# MCP Agent Demo

A small demo of an AI agent that calls tools over the Model Context Protocol (MCP). A Gradio app exposes four Python functions as an MCP server, and a smolagents `CodeAgent` behind a Gradio chat UI discovers and calls those tools to answer user requests.

## Architecture

```
Chat UI -> CodeAgent -> MCP client -> Gradio MCP server -> tools
```

- `mcp_server.py` defines the tools and launches a Gradio `TabbedInterface` with `mcp_server=True`, which serves them over MCP (SSE at `http://127.0.0.1:7860/gradio_api/mcp/sse`).
- `mcp_client.py` connects to that endpoint with smolagents' `MCPClient`, hands the tools to a `CodeAgent` backed by an OpenAI model, and wraps it in a Gradio `ChatInterface`.

## Tools

| Tool | What it does | Backed by |
| --- | --- | --- |
| `sentiment_analysis` | Returns polarity, subjectivity, and a positive/negative/neutral assessment as JSON | TextBlob |
| `summarize_text` | Summarizes text into concise paragraphs | OpenAI |
| `translate_to_french` | Translates text to French | OpenAI |
| `fetch_stock_price` | Returns the latest quote for a ticker symbol as JSON | Alpha Vantage `GLOBAL_QUOTE` |

## Setup

1. Install dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root (see `.env.example`):

   ```
   OPENAI_API_KEY=your-openai-key
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
   ```

   Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`).

3. Start the MCP server:

   ```bash
   python mcp_server.py
   ```

   It runs on port 7860, where you can also try each tool in its own tab.

4. In a second terminal, start the chat client:

   ```bash
   python mcp_client.py
   ```

   Open the URL it prints (usually `http://127.0.0.1:7861`) and ask something like *"Analyze the sentiment of: 'This is awesome'"* or *"Fetch the latest stock price for MSFT."*

## Screenshot

![Chat UI](docs/chat-ui.png)
