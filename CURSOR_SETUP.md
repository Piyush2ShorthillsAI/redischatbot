# Cursor MCP Redis Setup Guide

This directory contains configuration files for integrating the Redis MCP Server with Cursor.

## Configuration Files

### 📁 `cursor-mcp-config-local.json`
Configuration for **local Redis** development setup (Redis running on localhost:6379)

### 📁 `cursor-mcp-config-cloud.json` 
Configuration for **Redis Cloud** production setup (requires Redis Cloud credentials)

## How to Use

### Option 1: Local Redis Setup (Recommended for Development)

1. **Ensure Redis is running locally:**
   ```bash
   # Install Redis (Ubuntu/Debian)
   sudo apt update && sudo apt install redis-server
   sudo systemctl start redis-server
   
   # Test connection
   redis-cli ping  # Should return PONG
   ```

2. **Use the local configuration:**
   - Copy contents of `cursor-mcp-config-local.json`
   - Add to your Cursor MCP settings

### Option 2: Redis Cloud Setup (Recommended for Production)

1. **Get Redis Cloud credentials:**
   - Sign up at [Redis Cloud](https://redis.com/try-free/)
   - Create a database and note: endpoint, port, password

2. **Update the cloud configuration:**
   - Edit `cursor-mcp-config-cloud.json`
   - Replace placeholders:
     - `your-redis-cloud-endpoint.com` → your actual endpoint
     - `your-redis-cloud-password` → your actual password
     - Port may need adjustment (commonly 16379 for Redis Cloud)

3. **Use the updated configuration:**
   - Copy contents of the modified `cursor-mcp-config-cloud.json`
   - Add to your Cursor MCP settings

## Adding to Cursor

1. Open Cursor Settings (Ctrl/Cmd + ,)
2. Search for "MCP" or "Model Context Protocol"
3. Add the JSON configuration from your chosen file
4. Restart Cursor to load the MCP server

## Testing the Setup

Once configured, you can test by asking Cursor:
- "Store a test value in Redis"
- "Get Redis server information"
- "Set a key-value pair in Redis"

## Available Redis Operations

- **String operations**: Set, get, cache values
- **Hash operations**: Store objects and embeddings  
- **List operations**: Queues and sequences
- **Set operations**: Unique collections
- **Sorted sets**: Leaderboards and rankings
- **Streams**: Event logging
- **Pub/Sub**: Real-time messaging
- **JSON**: Complex documents
- **Vector search**: AI embeddings and similarity

## Troubleshooting

- **Connection failed**: Check Redis is running (`redis-cli ping`)
- **Permission errors**: Ensure file paths are correct
- **MCP not loading**: Restart Cursor after configuration changes
- **SSL issues**: Verify Redis Cloud SSL settings 