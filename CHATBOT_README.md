# Redis Vector Chatbot

A powerful chatbot that combines Redis Cloud vector storage with intelligent LLM-powered MCP tool selection for semantic search across your uploaded data.

## Features

🔍 **Intelligent Query Processing**: LLM analyzes user queries and automatically selects appropriate MCP tools
📤 **Multi-format Data Upload**: Support for TXT, JSON, CSV, and PDF files
🤖 **Smart Text Chunking**: Automatic text segmentation with overlap for better search results  
☁️ **Redis Cloud Integration**: Uses Redis Cloud with vector search capabilities
🔧 **MCP Tools Integration**: Leverages Model Context Protocol tools for Redis operations
🎨 **Modern UI**: Clean Streamlit interface with real-time chat and progress tracking

## Architecture

```
User Query → LLM Intent Analysis → MCP Tool Selection → Redis Vector Search → Context Generation → Response
```

### Key Components

1. **RedisVectorManager**: Handles Redis Cloud connections, vector operations, and data storage
2. **IntelligentChatbot**: LLM-powered agent that analyzes queries and calls appropriate MCP tools
3. **MCP Tools**: Redis operations including vector indexing, storage, and search
4. **StreamlitApp**: User interface for data upload and chat interaction

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `env_template.txt` to `.env` and configure your credentials:

```bash
cp env_template.txt .env
```

Edit `.env` with your actual credentials:

**For OpenAI:**
```
OPENAI_API_KEY=sk-your_actual_openai_key
```

**For Azure OpenAI (recommended):**
```
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

**For Redis Cloud:**
```
REDIS_URL=rediss://username:password@host:port/db
```

### 3. Redis Cloud Setup

1. Create a Redis Cloud account at [redis.com](https://redis.com)
2. Create a new database with **RediSearch** module enabled
3. Get your connection details (host, port, password)
4. Use the connection URL format: `rediss://username:password@host:port/db`

### 4. Run the Application

```bash
streamlit run streamlit_chatbot.py
```

## Usage Guide

### Uploading Data

#### Manual Input
- Paste any text content directly into the text area
- The system automatically chunks large text for optimal search

#### File Upload
- **TXT files**: Plain text content
- **JSON files**: Structured data (converted to searchable text)
- **CSV files**: Tabular data (each row becomes searchable)
- **PDF files**: Extracted text content with paragraph chunking

### Chatting with Your Data

The chatbot intelligently handles different types of queries:

**Data Search Queries:**
- "What information do you have about X?"
- "Find documents related to Y"
- "Tell me about Z"

**Redis Statistics:**
- "How many documents are stored?"
- "What's the status of the vector index?"
- "Show me database statistics"

**General Questions:**
- The LLM can also answer general questions and provide guidance

### How It Works

1. **Query Analysis**: The LLM analyzes your query to determine intent
2. **Tool Selection**: Based on intent, appropriate MCP tools are selected
3. **Vector Search**: If searching data, embeddings are created and vector search is performed
4. **Context Assembly**: Results are formatted and prepared for response generation
5. **Response Generation**: The LLM generates a helpful response based on found data

## MCP Tools Used

The chatbot leverages these MCP tools for Redis operations:

- `create_vector_index_hash`: Initialize vector search index
- `set_vector_in_hash`: Store document vectors
- `vector_search_hash`: Perform semantic similarity search
- `hset`: Store document metadata and content
- `scan_all_keys`: Get database statistics
- `get_index_info`: Retrieve index information

## Troubleshooting

### Common Issues

**Redis Connection Failed:**
- Verify your Redis Cloud credentials
- Ensure RediSearch module is enabled on your database
- Check if firewall allows connections

**OpenAI API Errors:**
- Verify API key is correct and has sufficient credits
- For Azure OpenAI, ensure endpoint and deployment names are correct

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that the `src/` directory is in your Python path

**No Search Results:**
- Make sure you've uploaded some data first
- Try different query phrasings
- Check if the vector index was created successfully

### Performance Tips

- **Chunk Size**: Default 1000 characters works well for most content
- **Overlap**: 200 character overlap helps maintain context across chunks
- **Query Specificity**: More specific queries often yield better results
- **Data Quality**: Clean, well-formatted input data improves search accuracy

## Advanced Configuration

### Custom Embedding Models

For Azure OpenAI, specify a different embedding deployment:
```
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### Redis Parameters

For fine-tuned Redis configuration:
```
REDIS_HOST=your_host
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_SSL=true
```

### Vector Index Settings

The system uses these defaults (configurable in code):
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: COSINE similarity
- **Vector Dimension**: 1536 (OpenAI ada-002 embedding size)
- **Index Prefix**: `doc:` for document keys

## Example Use Cases

1. **Document Search**: Upload company documents and ask questions about policies, procedures
2. **Knowledge Base**: Create a searchable knowledge base from manuals, guides, FAQs
3. **Research Assistant**: Upload research papers and ask for specific information
4. **Code Documentation**: Upload code documentation and ask about implementations
5. **Customer Support**: Upload support articles and get instant answers to common issues

## Contributing

This chatbot is built on the MCP Redis project. The main components are:

- `streamlit_chatbot.py`: Main application file
- `src/tools/`: MCP tool implementations  
- `src/common/`: Redis connection and configuration management

Feel free to extend the functionality by:
- Adding new MCP tools
- Implementing additional file format support
- Enhancing the LLM prompt strategies
- Adding more sophisticated chunking algorithms

## License

This project follows the same license as the base MCP Redis project. 