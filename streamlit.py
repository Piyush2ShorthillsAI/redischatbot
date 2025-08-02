import streamlit as st
import json
import openai
import numpy as np
from typing import List, Dict, Any, Optional
import os
import io
import uuid
import pandas as pd
from dotenv import load_dotenv
import asyncio
from PyPDF2 import PdfReader
import inspect

# Import MCP tools
import sys
sys.path.append('src')
from src.common.connection import RedisConnectionManager
from src.common.config import REDIS_CFG, set_redis_config_from_cli
from src.tools.redis_query_engine import create_vector_index_hash, vector_search_hash, get_index_info
from src.tools.hash import hset, hgetall, set_vector_in_hash
from src.tools.misc import scan_all_keys
from redis_function_mappings import function_mappings

# Functions from the 'hash.py' module
from src.tools.hash import hset, hgetall, hget, hdel, hexists, set_vector_in_hash, get_vector_from_hash

# Functions from the 'json.py' module
from src.tools.json import json_set, json_get, json_del

# Functions from the 'list.py' module
from src.tools.list import lpush, rpush, lpop, rpop, lrange, llen

# Functions from the 'misc.py' module
from src.tools.misc import scan_all_keys, scan_keys, delete, type, expire, rename
from src.tools.server_management import dbsize, info, client_list

# Functions from the 'pub_sub.py' module
from src.tools.pub_sub import publish, subscribe, unsubscribe

# Functions from the 'redis_query_engine.py' module
from src.tools.redis_query_engine import create_vector_index_hash, vector_search_hash, get_index_info, get_indexes, get_indexed_keys_number

# Functions from the 'set.py' module
from src.tools.set import sadd, srem, smembers

# Functions from the 'sorted_set.py' module
from src.tools.sorted_set import zadd, zrange, zrem

# Functions from the 'stream.py' module
from src.tools.stream import xadd, xrange, xdel

# Set page config FIRST
st.set_page_config(
    page_title="Redis Vector Chatbot",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Load tools from a JSON file
try:
    with open('redis_mcp_tools_openai.json', 'r') as f:
        OA_TOOLS = json.load(f)
except FileNotFoundError:
    st.error("❌ 'redis_mcp_tools_openai.json' not found. Please create this file with your tool definitions.")
    OA_TOOLS = []
except json.JSONDecodeError:
    st.error("❌ 'redis_mcp_tools_openai.json' is not a valid JSON file.")
    OA_TOOLS = []


class RedisVectorManager:
    """Manages Redis vector operations using MCP tools"""
    
    def __init__(self):
        # Setup OpenAI client
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.openai_client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            )
            self.use_azure = True
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", self.deployment_name)
        else:
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_azure = False
            self.deployment_name = "gpt-4o-mini"
            self.embedding_deployment = "text-embedding-ada-002"

        # Setup Redis Cloud connection
        self._setup_redis_connection()
        
        self.index_name = "vector_index"
        self.vector_field = "vector"
        self.dimension = 1536
        
        # Initialize vector index
        self._initialize_vector_index()

    def _setup_redis_connection(self):
        """Setup Redis Cloud connection using environment variables"""
        try:
            redis_config = {}
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                from src.common.config import parse_redis_uri
                redis_config = parse_redis_uri(redis_url)
            else:
                redis_config = {
                    'host': os.getenv('REDIS_HOST', 'localhost'),
                    'port': int(os.getenv('REDIS_PORT', 6379)),
                    'password': os.getenv('REDIS_PASSWORD', ''),
                    'username': os.getenv('REDIS_USERNAME'),
                    'ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            
            set_redis_config_from_cli(redis_config)
            
            conn = RedisConnectionManager.get_connection()
            conn.ping()
            st.success("✅ Connected to Redis Cloud successfully!")
            
        except Exception as e:
            st.error(f"❌ Redis connection failed: {str(e)}")
            st.info("Please check your Redis Cloud credentials in environment variables")

    def _initialize_vector_index(self):
        """Initialize vector index using MCP tool"""
        try:
            result = asyncio.run(create_vector_index_hash(
                index_name=self.index_name,
                prefix="doc:",
                vector_field=self.vector_field,
                dim=self.dimension,
                distance_metric="COSINE"
            ))
            if "successfully" in result or "already exists" in result.lower():
                st.info(f"🔧 Vector index '{self.index_name}' ready")
            else:
                st.warning(f"Index creation result: {result}")
        except Exception as e:
            st.warning(f"Index initialization: {str(e)}")

    def create_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                break_point = max(last_period, last_space)
                if break_point > start + chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk.strip()]

    def process_and_store_data(self, chunks: List[str], metadata: Optional[Dict] = None) -> bool:
        """Process and store text chunks with embeddings"""
        try:
            metadata = metadata or {}
            doc_id = metadata.get("doc_id", str(uuid.uuid4()))
            success_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                progress_bar.progress((i + 1) / len(chunks))
                
                embedding = self.create_embeddings(chunk)
                if not embedding:
                    continue
                
                doc_key = f"doc:{doc_id}:chunk:{i}"
                chunk_metadata = {**metadata, "doc_id": doc_id, "chunk_id": i, "content": chunk}
                
                try:
                    for key, value in chunk_metadata.items():
                        asyncio.run(hset(doc_key, key, str(value)))
                    
                    vector_stored = asyncio.run(set_vector_in_hash(doc_key, embedding, self.vector_field))
                    
                    if vector_stored:
                        success_count += 1
                        
                except Exception as e:
                    st.warning(f"Failed to store chunk {i}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if success_count > 0:
                st.success(f"✅ Successfully stored {success_count}/{len(chunks)} chunks")
                return True
            else:
                st.error("❌ Failed to store any chunks")
                return False
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return False

    def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform vector search using MCP tools"""
        try:
            query_embedding = self.create_embeddings(query)
            if not query_embedding:
                return []
            
            results = asyncio.run(vector_search_hash(
                query_vector=query_embedding,
                index_name=self.index_name,
                vector_field=self.vector_field,
                k=k,
                return_fields=["content", "doc_id", "chunk_id"]
            ))
            
            if isinstance(results, str):
                st.error(f"Search error: {results}")
                return []
                
            return results
            
        except Exception as e:
            st.error(f"Error performing vector search: {str(e)}")
            return []


class RedisExplorer:
    """Utility class to safely explore Redis database contents"""
    
    @staticmethod
    def explore_key_safely(key_name: str) -> dict:
        """Safely explore a Redis key by checking its type first"""
        try:
            # First, check if the key exists and get its type
            key_type = asyncio.run(type(key_name))
            
            result = {
                "key": key_name,
                "type": key_type,
                "data": None,
                "error": None
            }
            
            if key_type == "none":
                result["error"] = "Key does not exist"
                return result
            
            # Based on the type, use the appropriate retrieval method
            if key_type == "hash":
                result["data"] = asyncio.run(hgetall(key_name))
            elif key_type == "string":
                result["data"] = f"String value (use appropriate string retrieval method)"
            elif key_type == "list":
                # Get list length and some elements
                length = asyncio.run(llen(key_name))
                elements = asyncio.run(lrange(key_name, 0, min(9, length-1)))  # Get first 10 elements
                result["data"] = {
                    "length": length,
                    "sample_elements": elements
                }
            elif key_type == "set":
                members = asyncio.run(smembers(key_name))
                result["data"] = {
                    "members": list(members) if hasattr(members, '__iter__') else members
                }
            elif key_type == "zset":
                # Get sorted set elements
                elements = asyncio.run(zrange(key_name, 0, 9))  # Get first 10 elements
                result["data"] = {
                    "sample_elements": elements
                }
            else:
                result["data"] = f"Unsupported type: {key_type}"
            
            return result
            
        except Exception as e:
            return {
                "key": key_name,
                "type": "unknown",
                "data": None,
                "error": str(e)
            }
    
    @staticmethod
    def explore_database_overview() -> dict:
        """Get an overview of the Redis database"""
        try:
            # Get all keys with different patterns
            all_keys = list(asyncio.run(scan_all_keys("*")))
            
            # Group keys by type
            type_summary = {}
            sample_keys_by_type = {}
            
            for key in all_keys[:50]:  # Limit to first 50 keys to avoid overwhelming
                try:
                    key_type = asyncio.run(type(key))
                    if key_type not in type_summary:
                        type_summary[key_type] = 0
                        sample_keys_by_type[key_type] = []
                    
                    type_summary[key_type] += 1
                    if len(sample_keys_by_type[key_type]) < 5:
                        sample_keys_by_type[key_type].append(key)
                        
                except Exception as e:
                    print(f"Error checking type for key {key}: {e}")
                    continue
            
            return {
                "total_keys_scanned": len(all_keys),
                "type_summary": type_summary,
                "sample_keys_by_type": sample_keys_by_type,
                "note": "Limited to first 50 keys for performance"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to explore database: {str(e)}"
            }


class IntelligentChatbot:
    """Chatbot with LLM that can call MCP tools intelligently with type safety"""
    
    def __init__(self, redis_manager: RedisVectorManager, system_prompt: str):
        self.redis_manager = redis_manager
        self.openai_client = redis_manager.openai_client
        self.system_prompt = system_prompt
    
    def safe_tool_call(self, function_name: str, function_to_call, function_args: dict, query: str = None):
        """Safely call a tool with proper type checking and error handling"""
        try:
            # Special handling for type-sensitive operations
            if function_name in ['hgetall', 'hget', 'hset', 'hdel', 'hexists']:
                # Check if the key exists and is a hash before performing hash operations
                key_name = function_args.get('name') or function_args.get('key')
                if key_name:
                    # First check the type of the key
                    key_type = asyncio.run(type(key_name))
                    if key_type != 'hash' and key_type != 'none':
                        return f"Error: Key '{key_name}' is of type '{key_type}', not a hash. Cannot perform hash operations."
                    elif key_type == 'none':
                        return f"Error: Key '{key_name}' does not exist."
            
            # Handle different function types
            if function_name == "vector_search_hash":
                # Need to pass the user query for embedding
                function_response = asyncio.run(function_to_call(query))
            elif asyncio.iscoroutinefunction(function_to_call):
                # This is an async function, await it
                function_response = asyncio.run(function_to_call(**function_args))
            else:
                # This is a regular function, call it normally
                function_response = function_to_call(**function_args)
            
            # Ensure the response is JSON serializable
            if hasattr(function_response, '__iter__') and not isinstance(function_response, (str, dict)):
                # Convert iterables (like generators) to lists
                function_response = list(function_response)
            
            return function_response
            
        except Exception as e:
            error_msg = str(e)
            if "WRONGTYPE" in error_msg:
                key_name = function_args.get('name') or function_args.get('key', 'unknown')
                return f"Error: Key '{key_name}' is not the correct data type for operation '{function_name}'. Use the 'type' tool to check the key type first."
            return f"Error calling {function_name}: {error_msg}"
    
    def chat(self, query: str) -> str:
        """Main chat method that orchestrates the entire flow using tool-calling"""
        messages = [{"role": "system", "content": self.system_prompt}]
        # Add the user's query to the messages
        messages += st.session_state.chat_history + [{"role": "user", "content": query}]
        
        try:
            # Step 1: Send messages and tools to the LLM
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=OA_TOOLS,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            print(f"Tool calls: {tool_calls}")
            
            # Step 2: Check if the LLM wants to call a tool
            if tool_calls:
                messages.append(response_message)
                
                # Step 3: Call the tool and append the result
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = function_mappings.get(function_name)
                    print(f"Function to call: {function_name}")
                    
                    if function_to_call:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"Function args: {function_args}")
                            
                            # Use the safe tool call method
                            function_response = self.safe_tool_call(
                                function_name, function_to_call, function_args, query
                            )
                            
                            print(f"Function response: {function_response}")
                            
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(function_response, default=str)
                            })
                            
                        except Exception as e:
                            print(f"Error calling tool '{function_name}': {e}")
                            st.error(f"Error calling tool '{function_name}': {e}")
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {str(e)}"
                            })

                # Step 4: Get a final response from the LLM with tool output
                final_response = self.openai_client.chat.completions.create(
                    model=self.redis_manager.deployment_name,
                    messages=messages,
                )
                return final_response.choices[0].message.content
            else:
                # No tool call, just a regular conversational response
                return response_message.content
        
        except Exception as e:
            st.error(f"Error processing chat: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


# Enhanced system prompt with type safety
ENHANCED_SYSTEM_PROMPT = """
Redis Database Assistant

You are an intelligent assistant that interacts with a Redis database using specialized tools. Follow these rules strictly:

CORE PRINCIPLES:
- Respond ONLY with information retrieved from the connected Redis database using tools
- Do NOT use training data or general knowledge
- Always search Redis before answering; never assume data is missing without checking
- Ignore queries unrelated to stored Redis data

CRITICAL TYPE SAFETY RULES:
- ALWAYS check the data type of a key using the 'type' tool before performing operations
- Different Redis data types require different tools:
  * hash: use hgetall, hget, hset, hdel, hexists
  * string: use appropriate string operations  
  * list: use lpush, rpush, lpop, rpop, lrange, llen
  * set: use sadd, srem, smembers
  * zset (sorted set): use zadd, zrange, zrem
  * stream: use xadd, xrange, xdel

TOOL USAGE WORKFLOW:
1. First, use 'scan_all_keys' or 'scan_keys' to discover relevant keys
2. Use 'type' tool to check each key's data type
3. Based on the type, use the appropriate retrieval tool
4. Chain tools when needed for complex queries
5. Always handle errors gracefully

SEARCH STRATEGY:
- Use vector_search_hash for semantic/content-based queries
- Use scan_keys with patterns for key name-based searches
- Use type checking before any data retrieval
- Explore both key names and stored content

ERROR HANDLING:
- If you get "WRONGTYPE" errors, it means you used the wrong tool for that data type
- Always check key type first to avoid these errors
- Provide helpful error messages explaining the correct approach

Always base answers solely on Redis data you retrieved using the appropriate tools for each data type.
"""


class StreamlitApp:
    """Main Streamlit application with enhanced features"""
    
    def __init__(self):
        if "redis_manager" not in st.session_state:
            st.session_state.redis_manager = RedisVectorManager()
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = IntelligentChatbot(st.session_state.redis_manager, ENHANCED_SYSTEM_PROMPT)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "openai_messages" not in st.session_state:
            st.session_state.openai_messages = []

    def render_env_setup(self):
        """Show environment setup instructions"""
        with st.sidebar:
            st.header("🔧 Environment Setup")
            
            st.subheader("Required Environment Variables:")
            st.code("""
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
# OR for Azure OpenAI:
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Redis Cloud Configuration
REDIS_URL=redis://username:password@host:port/db
# OR individual parameters:
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_USERNAME=your_username
REDIS_SSL=true
            """)
            
            st.subheader("MCP Tools Used:")
            st.markdown("""
            - Vector Index Creation
            - Vector Storage & Retrieval  
            - Semantic Search
            - Data Scanning & Statistics
            - Type-Safe Operations
            """)

    def render_data_upload(self):
        """Render data upload interface"""
        st.header("📤 Upload Data")
        
        tab_text, tab_file = st.tabs(["Manual Input", "File Upload"])
        
        with tab_text:
            st.subheader("Enter Text Data")
            text_input = st.text_area("Enter your text data:", height=200, 
                                     placeholder="Paste any text content here...")
            
            if st.button("Store Text Data", type="primary"):
                if text_input.strip():
                    chunks = st.session_state.redis_manager.chunk_text(text_input)
                    success = st.session_state.redis_manager.process_and_store_data(
                        chunks, {"source": "manual_input", "type": "text"}
                    )
                    if success:
                        st.balloons()
                else:
                    st.warning("Please enter some text data.")
        
        with tab_file:
            st.subheader("Upload Files")
            uploaded_file = st.file_uploader(
                "Choose a file", 
                type=["txt", "json", "csv", "pdf"],
                help="Supported formats: TXT, JSON, CSV, PDF"
            )
            
            if uploaded_file:
                try:
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    content = ""
                    
                    if file_type == "txt":
                        content = uploaded_file.read().decode("utf-8")
                    elif file_type == "json":
                        json_data = json.load(uploaded_file)
                        content = json.dumps(json_data, indent=2)
                    elif file_type == "csv":
                        df = pd.read_csv(uploaded_file)
                        content = df.to_string()
                    elif file_type == "pdf":
                        reader = PdfReader(uploaded_file)
                        content = "\n".join([page.extract_text() for page in reader.pages])
                    
                    st.success(f"✅ File loaded: {len(content)} characters")
                    
                    if st.button("Process & Store File", type="primary"):
                        chunks = st.session_state.redis_manager.chunk_text(content)
                        success = st.session_state.redis_manager.process_and_store_data(
                            chunks, {
                                "source": "file_upload", 
                                "filename": uploaded_file.name,
                                "type": file_type
                            }
                        )
                        if success:
                            st.balloons()
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    def render_database_explorer(self):
        """Render database exploration interface"""
        st.header("🔍 Database Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Get Database Overview"):
                with st.spinner("Exploring database..."):
                    overview = RedisExplorer.explore_database_overview()
                    st.json(overview)
        
        with col2:
            key_to_explore = st.text_input("Explore specific key:")
            if st.button("Explore Key") and key_to_explore:
                with st.spinner(f"Exploring key: {key_to_explore}"):
                    key_info = RedisExplorer.explore_key_safely(key_to_explore)
                    st.json(key_info)

    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Intelligent Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask me anything about your data..."):
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.openai_messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking and searching..."):
                    response = st.session_state.chatbot.chat(query)
                st.write(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.openai_messages.append({"role": "assistant", "content": response})

    def render_sidebar_actions(self):
        """Render sidebar actions"""
        with st.sidebar:
            st.header("🚀 Actions")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.openai_messages = []
                st.rerun()
            
            if st.button("Check Redis Stats"):
                try:
                    # Properly await the async functions
                    keys = asyncio.run(scan_all_keys("doc:*"))
                    
                    # Convert to list if it's a generator or other iterable
                    if hasattr(keys, '__iter__') and not isinstance(keys, (str, dict, list)):
                        keys = list(keys)
                    elif not isinstance(keys, list):
                        keys = []
                    
                    key_count = len(keys)
                    
                    # Await the result of get_index_info
                    index_info = asyncio.run(get_index_info(st.session_state.redis_manager.index_name))
                    
                    # Ensure index_info is JSON-serializable
                    if not isinstance(index_info, dict):
                        index_info = {"info": str(index_info)}

                    # Display the results
                    st.success(f"📊 Stats:\n- Documents: {key_count}\n- Index: {st.session_state.redis_manager.index_name}")
                    st.json(index_info)

                except Exception as e:
                    print(f"Error in stats: {str(e)}")
                    st.error(f"Error getting stats: {str(e)}")

    def run(self):
        """Run the application"""
        st.title("🔍 Redis Vector Chatbot")
        st.markdown("Upload any data and chat with it using semantic search powered by Redis Cloud + MCP tools")
        
        self.render_env_setup()
        self.render_sidebar_actions()
        
        tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "💬 Chat", "🔍 Database Explorer"])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_chat_interface()
        
        with tab3:
            self.render_database_explorer()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
    