import os
import re
from typing import List, Optional, Dict
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph, Neo4jVector
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

class UserFileFilterSearch(BaseModel):
    # username: Annotated[str, InjectedToolArg] = Field(
    #    description="Username filter, mandatory to check only for current username"
    # )
    state: Annotated[dict, InjectedState] = Field(
        description="Current conversation state"
    )
    filter_file_name: Optional[List[str]] = Field(
        description="List of file names for which uploaded files will be picked",
        default=None
    )
    filter_date_from: Optional[str] = Field(
        description="Files uploaded starting from this date (including this date)",
        default=None
    )
    filter_date_till: Optional[str] = Field(
        description="Files uploaded till this date (including this date)",
        default=None
    )
    similarity_search_message: Optional[str] = Field(
        description="Message/keywords for which similarity search will be performed on file chunks",
        default=None
    )
    limit_by: Optional[int] = Field(
        description="limits number of output records, maximum is 10",
        default=4,
        le=10
    )
    show_image: Optional[bool] = Field(
        description="If filter_file_name is used passing True for this will show the image on UI. This will work only "
                    "with jpeg/png/pdf files",
        default=False
    )


@tool("file-filter-search", args_schema=UserFileFilterSearch)
def file_filter_search(
        state: Annotated[dict, InjectedState],
        filter_file_name: Optional[List[str]] = None,
        filter_date_from: Optional[str] = None,
        filter_date_till: Optional[str] = None,
        similarity_search_message: Optional[str] = None,
        limit_by: Optional[int] = 4,
        show_image: Optional[bool] = False
) -> Dict:
    """This tool allows agents to search for current or previously uploaded files by the user. It supports:
    - Date Range Search: Use from_date and to_date (in UTC) to retrieve up to 4 of the latest files within that range.
    - Filename Search: Provide a list of filenames to fetch full file contents directly.
    - Similarity Search: Use similarity_search_message with keywords/phrases to find contextually similar files. This can be
        combined with date filters to narrow results.
    - Time Interpretation: User message timestamps are in UTC. Use them to resolve relative time references (e.g., "last week")
        into specific date ranges.
    - Image Retrieval: For JPEG, PNG, or PDF files, set show_image=True along with the filename to display images from
        the file. User may ask for the image, use this feature in that case.
    Input Parameters:
    1. filter_file_name - (optional) List of file names for which uploaded files will be picked. If you know exact file
        names and need full contents use this parameter. This is also needed when user wants images to be shown on UI.
    2. filter_date_from - (optional) Files uploaded starting from this date (including this date). Format: yyyy-MM-dd
        Example: 2025-01-31
    3. filter_date_till - (optional) Files uploaded till this date (including this date). Format: yyyy-MM-dd
        Example: 2025-02-15
    4. similarity_search_message - (optional) A text statement/phrase/set of keywords for which similarity search will
        be performed. This is optional, if you need all previous files or files with specific names don't use this. For
        example if user has uploaded file1.pdf and asking something from the file only use file name filter and keep this empty.
    5. limit_by - (optional) Defines how many messages can be fetched. By default, it will be 4 and at most it can be 10
    6. show_image - (optional) If filter_file_name is used to get specific files, passing this True will show the
        image on UI. This will work only with jpeg/png/pdf files. If user specifically asks to show/display any file
        abc.png and xyz.png use this True with filter_file_name = ['abc.png', 'xyz.png'].
        You do not need to use ![chart]() format for this.
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

    # Normal neo4j search
    if similarity_search_message is None:
        if access_role == 'Admin':
            file_filter = f"WHERE 1=1 "
        elif access_role == 'User':
            file_filter = f"WHERE u.username = '{username}' "
        else:
            file_filter = f"WHERE u.username = '{username}' "

        if filter_file_name is not None:
            file_filter += f''' AND f.name IN [{",".join("'" + x + "'" for x in filter_file_name)}]'''

        if filter_date_from is None and filter_date_till is None:
            pass
        elif filter_date_from is not None:
            if not date_pattern.match(filter_date_from):
                raise ValueError('filter_date_from must be in yyyy-MM-dd format')
            file_filter += f" AND f.timestamp >= '{filter_date_from}'"
        elif filter_date_till is not None:
            if not date_pattern.match(filter_date_till):
                raise ValueError('filter_date_till must be in yyyy-MM-dd format')
            file_filter += f" AND f.timestamp <= '{filter_date_till}'"

        cypher_query = ("MATCH (u:User)-[:UPLOADED_FILE]->(f:File) "
                        + file_filter + " "
                                        "WITH u, f ORDER BY f.timestamp DESC "
                                        "LIMIT " + str(limit_by) + " "
                                                                   "WITH f ORDER BY f.timestamp DESC "
                                                                   "MATCH (f:File)-[:CHUNKED_INTO]->(c:Chunk) "
                                                                   "ORDER BY c.chunk_no ASC "
                                                                   "WITH f, COLLECT(c.text) AS texts "
                                                                   "RETURN f as file_details, REDUCE(s = '', p IN texts | s + ' ' + p) AS file_contents")

        graph_response = graph.query(cypher_query)
        return_dict = {}

        if show_image:
            return_dict['metadata'] = {}
            return_dict['metadata']['image_data'] = {res['file_details']['name']: res['file_details']['data'] for res in
                                                     graph_response if
                                                     'data' in res['file_details'].keys()}

        for res in graph_response:
            if 'data' in res['file_details'].keys():
                del res['file_details']['data']
        return_dict['readable'] = graph_response

        return return_dict

    # Neo4j similarity/Hybrid search
    else:
        filters = {'username': username}
        if filter_date_from is not None and filter_date_till is not None:
            if not date_pattern.match(filter_date_from):
                raise ValueError('filter_date_from must be in yyyy-MM-dd format')
            if not date_pattern.match(filter_date_till):
                raise ValueError('filter_date_from must be in yyyy-MM-dd format')
            filters['timestamp'] = {'$between': [filter_date_from, filter_date_till]}
        elif filter_date_from is not None:
            if not date_pattern.match(filter_date_from):
                raise ValueError('filter_date_from must be in yyyy-MM-dd format')
            filters['timestamp'] = {'$gte': filter_date_from}
        elif filter_date_till is not None:
            if not date_pattern.match(filter_date_till):
                raise ValueError('filter_date_from must be in yyyy-MM-dd format')
            filters['timestamp'] = {'$lte': filter_date_till}

        if filter_file_name is not None:
            filters['name'] = {'$in': filter_file_name}

        db = Neo4jVector(embedding=embeddings, node_label='Chunk', embedding_node_property='embedding')
        search_result = db.similarity_search_with_score(query=similarity_search_message, k=limit_by, filter=filters)

        return_dict = {'readable': []}
        for res in search_result:
            formatted_doc_dict = {
                "content": res[0].page_content,
                "chunk_no": res[0].metadata['chunk_no'],
                "origin_filename": res[0].metadata['origin_filename'],
                "chunk_create_ts": res[0].metadata['chunk_create_ts']
            }
            return_dict['readable'].append({"chunk": formatted_doc_dict, "similarity_score": res[1]})

        return return_dict
