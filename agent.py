from typing import Any, Generator, Optional, Sequence, Union
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph, MessagesState, START
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
import os
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from databricks.sdk import WorkspaceClient
import urllib.parse
from databricks_ai_bridge import ModelServingUserCredentials
from psycopg_pool import ConnectionPool

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = [
    "bo_cheng_dnb_demos.agents.get_cyber_threat_info",
    "bo_cheng_dnb_demos.agents.get_user_info",
]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
#
# # TODO: Add vector search indexes as tools or delete this block
# vector_search_index_tools = [
#     VectorSearchRetrieverTool(
#         index_name="bo_cheng_dnb_demos.agents.poc_customer_support_index",
#         num_results=3,
#         tool_name="customer_support_retriever",
#         tool_description="Retrieves information about customer support responses",
#         query_type="ANN",
#     )
# ]
# tools.extend(vector_search_index_tools)

#####################
## Define agent logic
#####################


class LangGraphChatAgent(ChatAgent):
    def __init__(self, config, tools):
        self.config = config
        # self.connstring = conn
        self.conn_db_name = self.config["conn_db_name"]
        self.conn_ssl_mode = self.config["conn_ssl_mode"]
        self.conn_host = self.config["conn_host"]
        self.instance_name = self.config["instance_name"]
        self.tools = tools
        self.model = ChatDatabricks(
            endpoint=self.config.get("llm_model_serving_endpoint_name"), temperature=0.1
        ).bind_tools(self.tools)
        self.system_prompt = self.config.get("llm_prompt_template")
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
        self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))

    def _get_oauth_connection_string(self):
        """Get a fresh OAuth token and return connection string"""
        # self.w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        self.w = WorkspaceClient()
        try:
            sp = self.w.current_service_principal.me()
            sp_username = sp.application_id
        except Exception:
            user = self.w.current_user.me()
            sp_username = urllib.parse.quote_plus(
                user.user_name
            )  # we need to allow encoding for local testing of the agent to work since it will pass in username

        pg_credential = self.w.database.generate_database_credential(
            request_id=str(uuid.uuid4()), instance_names=[self.instance_name]
        )

        conn_string = f"postgresql://{sp_username}:{pg_credential.token}@{self.conn_host}:5432/{self.conn_db_name}?sslmode={self.conn_ssl_mode}"

        return conn_string

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        try:
            thread_id = custom_inputs.get("thread_id")
        except:
            thread_id = str(uuid.uuid4())

        request = {"messages": self._convert_messages_to_dict(messages)}
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        messages = []
        if self.system_prompt:
            preprocessor = RunnableLambda(
                lambda state: [{"role": "system", "content": self.system_prompt}]
                + state["messages"]
            )
        else:
            preprocessor = RunnableLambda(lambda state: state["messages"])
        self.model = self.model.bind_tools(self.tools)
        model_runnable = preprocessor | self.model

        # Get connection string to connect to lakebase postgres instance
        conn_info = self._get_oauth_connection_string()

        # Run the agent with the checkpointer
        with ConnectionPool(
            conninfo=conn_info,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            timeout=self.pool_timeout,
            # Configure connection settings
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        ).connection() as conn:
            checkpointer = PostgresSaver(conn)

            def should_continue(state: ChatAgentState):
                messages = state["messages"]
                last_message = messages[-1]
                # If there are function calls, continue. else, end
                if last_message.get("tool_calls"):
                    return "continue"
                else:
                    return "end"

            def call_model(
                state: ChatAgentState,
                config: RunnableConfig,
            ):
                response = model_runnable.invoke(state, config)

                return {"messages": [response]}

            workflow = StateGraph(ChatAgentState)

            workflow.add_node("agent", RunnableLambda(call_model))
            workflow.add_node("tools", ChatAgentToolNode(self.tools))

            workflow.set_entry_point("agent")
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "continue": "tools",
                    "end": END,
                },
            )
            workflow.add_edge("tools", "agent")

            graph = workflow.compile(checkpointer=checkpointer)
            for event in graph.stream(
                request, checkpoint_config, stream_mode="updates"
            ):
                # print(event)
                for node_data in event.values():
                    messages.extend(
                        ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                    )
                # print(messages)
            return ChatAgentResponse(messages=messages)


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()

config = {
    "llm_model_serving_endpoint_name": "databricks-claude-3-7-sonnet",
    "llm_prompt_template": """
    You are an cybersecurity assistant.
    You are given a task and you must complete it.
    Use the following routine to support the customer.
    # Routine:
    1. Provide the get_cyber_threat_info tool the type of threat being asked about.
    2. Use the source ip address provided in step 1 as input for the get_user_info tool to retrieve user specific info.
    Use the following tools to complete the task:
    {tools}""",
    "conn_db_name": "databricks_postgres",
    "conn_ssl_mode": "require",
    "conn_host": "instance-71597a8a-7e99-4c85-b29a-f751a73ecb85.database.cloud.databricks.com",
    "instance_name": "bo-test-lakebase-3",
}

AGENT = LangGraphChatAgent(config=config, tools=tools)
mlflow.models.set_model(AGENT)
