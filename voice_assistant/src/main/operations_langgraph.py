import json
from functools import partial
import streamlit as st

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from tool_files_filter_search import file_filter_search
from tool_previous_chat_filter_search import previous_chat_filter_search

# Azure OpenAI Config
AZURE_OPENAI_MODEL = st.secrets['AZURE_OPENAI_MODEL']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_VERSION = st.secrets['AZURE_OPENAI_VERSION']


def assistant(state: MessagesState, sys_msg: SystemMessage, model):
    for messages in state["messages"]:
        if isinstance(messages, ToolMessage) and messages.name == 'file-based-data-retrieval':
            # print(messages.content)
            tool_response = json.loads(messages.content)
            if 'metadata' in tool_response.keys():
                del tool_response['metadata']
                messages.content = json.dumps(tool_response).replace('\\\\', '\\')

    # Dynamic trimming of longer messages in prompt
    tokens = 30000
    for i in range(len(state["messages"]) - 1, -1, -1):
        if isinstance(state["messages"][i].content, str):
            limit = int(tokens * 0.7)
            if len(state["messages"][i].content) > limit:
                state["messages"][i].content = state["messages"][i].content[
                                               :limit] + '...message is too long, truncating rest'
                tokens -= limit
            if isinstance(state["messages"][i], ToolMessage) and '"root_node":' in state["messages"][i].content:
                state["messages"][i].content = (
                        '' + state["messages"][i].content)  # add extra details before tool response

    messages = model.invoke([sys_msg] + state["messages"])
    # print(messages)
    return {"messages": [messages]}


def build_graph(logged_user_details):
    # Input validation
    if 'first_name' not in logged_user_details or 'username' not in logged_user_details or 'role' not in logged_user_details:
        raise Exception('Langgraph model cannot be built, missing mandatory user parameters')
    # Initialize the OpenAI LLM with Azure configuration
    model = AzureChatOpenAI(
        model=AZURE_OPENAI_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    # Create tools and bind with llm
    tools = [file_filter_search, previous_chat_filter_search]
    langgraph_model = model.bind_tools(tools)

    system_prompt = ('''You are a multilingual voice assistant specialized in reading and summarizing user-provided documents.
Your core tasks include summarizing content, answering questions based on document summaries, and performing deeper file 
searches when needed. Users can upload Word, Excel, PDF, PNG, or JPEG files via the "Browse files" button. You receive 
the file name and its summary.
- If the currently uploaded file summary lacks the answer, use the tool files-filter-search with file name to do full 
file search. If similarity search does not bring results Do this step by default instead of asking user for permission.
- Redirect users to document-related queries if they ask unrelated questions except asking for previous/older files or conversation.
- Access previous files and chats using provided tools and filters (name, date, or vector similarity).
- Interpret references like “this” or “last file” using chat context.
- If data is insufficient, politely inform the user.
- Support both voice and chat input/output. Use the file tool when asked to show images.
- If similarity search tools with similarity_search_message parameter is not retrieving results try again with just
filters like name of the file or date range only.
- Refer tool documentation on what each tool does and their parameter usage process. '''
                     + f"Current user name is {logged_user_details['first_name']} with access role {logged_user_details['role']}")

    sys_msg = SystemMessage(content=system_prompt)

    assistant_prefilled = partial(assistant, sys_msg=sys_msg, model=langgraph_model)
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_prefilled)
    builder.add_node("tools", ToolNode(tools))
    # Define edges:
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile(checkpointer=MemorySaver())
    return react_graph
