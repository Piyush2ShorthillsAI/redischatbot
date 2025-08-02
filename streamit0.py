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

# Import MCP tools
import sys
sys.path.append('src')
from src.common.connection import RedisConnectionManager
from src.common.config import REDIS_CFG, set_redis_config_from_cli
from src.tools.redis_query_engine import create_vector_index_hash, vector_search_hash, get_index_info
from src.tools.hash import hset, hgetall, set_vector_in_hash
from src.tools.misc import scan_all_keys

# Set page config FIRST
st.set_page_config(
    page_title="Redis Vector Chatbot",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
load_dotenv()

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
        else:
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_azure = False
            self.deployment_name = None

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
            # Configure Redis for cloud connection
            redis_config = {}
            
            # Check for Redis URL first (common in cloud services)
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                from src.common.config import parse_redis_uri
                redis_config = parse_redis_uri(redis_url)
            else:
                # Use individual parameters
                redis_config = {
                    'host': os.getenv('REDIS_HOST', 'localhost'),
                    'port': int(os.getenv('REDIS_PORT', 6379)),
                    'password': os.getenv('REDIS_PASSWORD', ''),
                    'username': os.getenv('REDIS_USERNAME'),
                    'ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            
            set_redis_config_from_cli(redis_config)
            
            # Test connection
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
            if self.use_azure:
                embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", self.deployment_name)
                response = self.openai_client.embeddings.create(
                    model=embedding_deployment,
                    input=text
                )
            else:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
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
            
            # Try to break at sentence or word boundary
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
                
                # Create embedding
                embedding = self.create_embeddings(chunk)
                if not embedding:
                    continue
                
                # Store document data and vector
                doc_key = f"doc:{doc_id}:chunk:{i}"
                chunk_metadata = {**metadata, "doc_id": doc_id, "chunk_id": i, "content": chunk}
                
                # Store metadata and content
                try:
                    # Store content and metadata
                    for key, value in chunk_metadata.items():
                        asyncio.run(hset(doc_key, key, str(value)))
                    
                    # Store vector
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
            # Create query embedding
            query_embedding = self.create_embeddings(query)
            if not query_embedding:
                return []
            
            # Perform vector search
            results = asyncio.run(vector_search_hash(
                query_vector=query_embedding,
                index_name=self.index_name,
                vector_field=self.vector_field,
                k=k,
                return_fields=["content", "doc_id", "chunk_id"]
            ))
            
            if isinstance(results, str):  # Error message
                st.error(f"Search error: {results}")
                return []
                
            return results
            
        except Exception as e:
            st.error(f"Error performing vector search: {str(e)}")
            return []

class IntelligentChatbot:
    """Chatbot with LLM that can call MCP tools intelligently"""
    
    def __init__(self, redis_manager: RedisVectorManager):
        self.redis_manager = redis_manager
        self.openai_client = redis_manager.openai_client
        
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine what MCP tools to use"""
        try:
            # Simplified analysis logic
            query_lower = query.lower()
            
            # Check for statistics/info requests
            if any(word in query_lower for word in ["how many", "count", "stats", "status", "info", "stored"]):
                return {
                    "intent": "get_stats",
                    "search_type": "none", 
                    "redis_info": "keys_count",
                    "reasoning": "User asking for database statistics"
                }
            
            # Check for data upload requests
            if any(word in query_lower for word in ["upload", "store", "add", "save"]):
                return {
                    "intent": "upload_data",
                    "search_type": "none",
                    "redis_info": "none", 
                    "reasoning": "User wants to upload data"
                }
            
            # Check if it's a search query (most common case)
            search_indicators = ["what", "tell me", "find", "search", "about", "information", "know"]
            if any(word in query_lower for word in search_indicators) or len(query.split()) > 2:
                return {
                    "intent": "search_data",
                    "search_type": "vector_search",
                    "redis_info": "none",
                    "reasoning": "User searching for information in stored data"
                }
            
            # Default to general question
            return {
                "intent": "general_question", 
                "search_type": "none",
                "redis_info": "none",
                "reasoning": "General conversation or unclear intent"
            }
            
        except Exception as e:
            st.error(f"Error analyzing query intent: {str(e)}")
            return {
                "intent": "general_question", 
                "search_type": "none", 
                "redis_info": "none",
                "reasoning": "Error in analysis, defaulting to general question"
            }

    def execute_mcp_actions(self, intent_analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Execute appropriate MCP tools based on intent analysis"""
        results = {"search_results": [], "redis_info": {}, "action_taken": "none"}
        
        try:
            if intent_analysis["intent"] == "search_data":
                if intent_analysis["search_type"] == "vector_search":
                    results["search_results"] = self.redis_manager.vector_search(query, k=5)
                    results["action_taken"] = "vector_search"
                    
            elif intent_analysis["intent"] == "get_stats":
                if intent_analysis["redis_info"] == "index_info":
                    info = asyncio.run(get_index_info(self.redis_manager.index_name))
                    results["redis_info"]["index_info"] = info
                    results["action_taken"] = "get_index_info"
                    
                elif intent_analysis["redis_info"] == "keys_count":
                    keys = asyncio.run(scan_all_keys("doc:*"))
                    results["redis_info"]["keys_count"] = len(keys) if isinstance(keys, list) else 0
                    results["action_taken"] = "scan_keys"
                    
        except Exception as e:
            st.error(f"Error executing MCP actions: {str(e)}")
            
        return results

    def generate_response(self, query: str, mcp_results: Dict[str, Any], intent_analysis: Dict[str, Any]) -> str:
        """Generate intelligent response based on query and MCP tool results"""
        try:
            # Prepare context from MCP results
            context = ""
            
            if mcp_results["search_results"]:
                context += "Search Results:\n"
                for i, result in enumerate(mcp_results["search_results"][:3]):
                    content = result.get("content", "N/A")
                    score = result.get("score", "N/A")
                    context += f"Result {i+1} (Score: {score}):\n{content}\n\n"
            
            if mcp_results["redis_info"]:
                context += "Redis Information:\n"
                for key, value in mcp_results["redis_info"].items():
                    context += f"{key}: {value}\n"
            
            # Generate response
            reasoning = intent_analysis.get("reasoning", "Query analysis completed")
            action_taken = mcp_results.get("action_taken", "none")
            
            response_prompt = f"""
            You are a helpful AI assistant with access to a Redis vector database.
            
            User Query: "{query}"
            Intent Analysis: {reasoning}
            Actions Taken: {action_taken}
            
            Available Context:
            {context}
            
            Provide a helpful, accurate response based on the available information. If no relevant data was found, explain this clearly and suggest how the user might get better results.
            """
            
            model = self.redis_manager.deployment_name if self.redis_manager.use_azure else "gpt-3.5-turbo"
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": response_prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def chat(self, query: str) -> str:
        """Main chat method that orchestrates the entire flow"""
        # Step 1: Analyze intent
        intent_analysis = self.analyze_query_intent(query)
        
        # Step 2: Execute appropriate MCP tools
        mcp_results = self.execute_mcp_actions(intent_analysis, query)
        
        # Step 3: Generate intelligent response
        response = self.generate_response(query, mcp_results, intent_analysis)
        
        return response

class StreamlitApp:
    """Main Streamlit application"""
    
    def __init__(self):
        if "redis_manager" not in st.session_state:
            st.session_state.redis_manager = RedisVectorManager()
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = IntelligentChatbot(st.session_state.redis_manager)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

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

    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Intelligent Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask me anything about your data..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking and searching..."):
                    response = st.session_state.chatbot.chat(query)
                st.write(response)
            
            # Add assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    def render_sidebar_actions(self):
        """Render sidebar actions"""
        with st.sidebar:
            st.header("🚀 Actions")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("Check Redis Stats"):
                try:
                    # Get some basic stats
                    keys = asyncio.run(scan_all_keys("doc:*"))
                    key_count = len(keys) if isinstance(keys, list) else 0
                    
                    index_info = asyncio.run(get_index_info(st.session_state.redis_manager.index_name))
                    
                    st.success(f"📊 Stats:\n- Documents: {key_count}\n- Index: {st.session_state.redis_manager.index_name}")
                    
                except Exception as e:
                    st.error(f"Error getting stats: {str(e)}")

    def run(self):
        """Run the application"""
        st.title("🔍 Redis Vector Chatbot")
        st.markdown("Upload any data and chat with it using semantic search powered by Redis Cloud + MCP tools")
        
        self.render_env_setup()
        self.render_sidebar_actions()
        
        # Main tabs
        tab1, tab2 = st.tabs(["📤 Upload Data", "💬 Chat"])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_chat_interface()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()