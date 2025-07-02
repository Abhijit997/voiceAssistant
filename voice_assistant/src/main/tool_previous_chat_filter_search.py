import os
import re
from typing import Dict, Optional
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_USER = st.secrets['NEO4J_USER']
NEO4J_PASSWORD = st.secrets['NEO4J_PASSWORD']
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

AZURE_EMBEDDING_MODEL = st.secrets['AZURE_EMBEDDING_MODEL']
AZURE_EMBEDDING_ENDPOINT = st.secrets['AZURE_EMBEDDING_ENDPOINT']
AZURE_EMBEDDING_KEY = st.secrets['AZURE_EMBEDDING_KEY']

class UserPreviousChatFilterSearch(BaseModel):
    state: Annotated[dict, InjectedState] = Field(
        description="Current conversation state"
    )
    filter_date_from: Optional[str] = Field(
        description="Chat messages to consider starting from this date (including this date)",
        default=None
    )
    filter_date_till: Optional[str] = Field(
        description="Chat messages to consider till this date (including this date)",
        default=None
    )
    similarity_search_message: Optional[str] = Field(
        description="Message/keywords for which similarity search will be performed on older chat messages",
        default=None
    )
    limit_by: Optional[int] = Field(
        description="limits number of output records, maximum is 10",
        default=10,
        le=10
    )


@tool("previous-chat-filter-search", args_schema=UserPreviousChatFilterSearch)
def previous_chat_filter_search(
        state: Annotated[dict, InjectedState],
        filter_date_from: Optional[str] = None,
        filter_date_till: Optional[str] = None,
        similarity_search_message: Optional[str] = None,
        limit_by: Optional[int] = 10
) -> Dict:
    """This tool allows agents to search past conversations with the current user. It supports:
    - Date Range Search: Use from_date and to_date (in UTC) to retrieve up to 10 recent messages within that range.
    - Similarity Search: Use similarity_search_message with keywords/phrases to find contextually similar past
        conversations. Can be combined with date filters for more precise results.
    - Time Interpretation: All dates must be in UTC. Use timestamps from user messages to resolve relative time
        references (e.g., "last week") into specific date ranges, possibly using two ranges for vague periods.
    Input Parameters:
    1. filter_date_from - (optional) Chat messages to consider starting from this date (including this date). Format:
        yyyy-MM-dd Example: 2025-01-31
    2. filter_date_till - (optional) Chat messages to consider till this date (including this date). Format:
        yyyy-MM-dd Example: 2025-02-15
    3. similarity_search_message - (optional) A text statement/phrase/set of keywords for which similarity search will
        be performed. This is optional, if you need all previous chat messages don't use this.
    4. limit_by - (optional) Defines how many messages can be fetched. By default, it will be 10 and at most it can be 10
    """
    embeddings = AzureOpenAIEmbeddings(
        model=AZURE_EMBEDDING_MODEL,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=AZURE_EMBEDDING_KEY,
    )
    if 'messages' not in state:
        raise Exception('Could not fetch current session state')

    # Fetch first HumanMessage
    human_message = next((msg for msg in state['messages'] if isinstance(msg, HumanMessage)), None)

    if human_message is None:
        raise Exception('Could not find human message state containing user metadata')
    if 'user_details' not in human_message.metadata:
        raise Exception('Could not find human message state containing user metadata')
    if 'username' not in human_message.metadata['user_details']:
        raise Exception('Could not find username from user metadata')

    username = human_message.metadata['user_details']['username']
    access_role = human_message.metadata['user_details']['role']
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    graph = Neo4jGraph()

    if access_role == 'Admin':
        chat_filter = f"WHERE 1=1 "
    elif access_role == 'User':
        chat_filter = f"WHERE u.username = '{username}' "
    else:
        chat_filter = f"WHERE u.username = '{username}' "

    if filter_date_from is None and filter_date_till is None:
        pass
    elif filter_date_from is not None:
        if not date_pattern.match(filter_date_from):
            raise ValueError('filter_date_from must be in yyyy-MM-dd format')
        chat_filter += f" AND c.timestamp >= '{filter_date_from}'"
    elif filter_date_till is not None:
        if not date_pattern.match(filter_date_till):
            raise ValueError('filter_date_till must be in yyyy-MM-dd format')
        chat_filter += f" AND c.timestamp <= '{filter_date_till}'"

    # Normal neo4j search
    if similarity_search_message is None:
        cypher_query = ("MATCH (u:User)-[:CONVERSED]->(c:Chat) "
                        + chat_filter + " "
                                        "WITH u, c ORDER BY c.timestamp DESC "
                                        "LIMIT " + str(limit_by) + " "
                                                                   "MATCH (c:Chat)-[FOLLOWED_BY*0..]->(d:Chat) "
                                                                   "WITH c.timestamp AS timestamp, {user_query: d.user_query, agent_response: d.agent_response} AS combined "
                                                                   "WITH timestamp, COLLECT(combined) AS chat_content "
                                                                   "RETURN timestamp, chat_content")

        return_dict = {'readable': graph.query(cypher_query)}
        return return_dict

    # Neo4j similarity/Hybrid search
    else:
        #   filters = {'username': username}
        #   if filter_date_from is not None and filter_date_till is not None:
        #       if not date_pattern.match(filter_date_from):
        #           raise ValueError('filter_date_from must be in yyyy-MM-dd format')
        #       if not date_pattern.match(filter_date_till):
        #           raise ValueError('filter_date_from must be in yyyy-MM-dd format')
        #       filters['timestamp'] = {'$between': [filter_date_from, filter_date_till]}
        #   elif filter_date_from is not None:
        #       if not date_pattern.match(filter_date_from):
        #           raise ValueError('filter_date_from must be in yyyy-MM-dd format')
        #       filters['timestamp'] = {'$gte': filter_date_from}
        #   elif filter_date_till is not None:
        #       if not date_pattern.match(filter_date_till):
        #           raise ValueError('filter_date_from must be in yyyy-MM-dd format')
        #       filters['timestamp'] = {'$lte': filter_date_till}
        #   db = Neo4jVector(embedding=embeddings, node_label='Chat', embedding_node_property='embedding')
        #   search_result = db.similarity_search_with_score(query=similarity_search_message, k=limit_by, filter=filters)

        embedding_vector = embeddings.embed_query(similarity_search_message)
        cypher_message = ("MATCH (u:User)-[:CONVERSED]->(c:Chat) "
                          + chat_filter + " "
                                          "WITH u, c, apoc.convert.fromJsonList(c.embedding) AS embedding "
                                          "WITH u, c, vector.similarity.cosine(embedding, " + str(
                    embedding_vector) + ") AS similarity_score "
                                        "ORDER BY similarity_score DESC LIMIT " + str(limit_by) + " "
                                                                                                  "WITH c AS chat, similarity_score "
                                                                                                  "MATCH (chat)-[FOLLOWED_BY*0..]->(d:Chat) "
                                                                                                  "WITH chat.timestamp AS timestamp, similarity_score, {user_query: d.user_query, agent_response: d.agent_response} AS single_chat "
                                                                                                  "WITH timestamp, similarity_score, COLLECT(single_chat) AS chat_flow "
                                                                                                  "RETURN timestamp, similarity_score, chat_flow ")

        search_result = graph.query(cypher_message)

        return_dict = {'readable': []}
        print(search_result)
        for res in search_result:
            formatted_doc_dict = {
                "full_chat": res['chat_flow'],
                "timestamp": res['timestamp']
            }
            return_dict['readable'].append({"chat": formatted_doc_dict, "similarity_score": res['similarity_score']})

        return return_dict

# print(embeddings_similarity_search.get_input_schema().model_json_schema())
# print(embeddings_similarity_search.tool_call_schema.model_json_schema())
