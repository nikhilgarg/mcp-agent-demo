# --- Imports ---
import gradio as gr # For creating the web-based chat UI
from smolagents import CodeAgent, MCPClient # CodeAgent executes tool-using code; MCPClient connects to the MCP server
import os
from smolagents.models import OpenAIServerModel, ChatMessage # For LLM interaction and message formatting

# Setup the client connection to MCP server
try:
    # Initialize the MCP client with the local server's SSE endpoint
    mcp_client = MCPClient({"url": "http://127.0.0.1:7860/gradio_api/mcp/sse"})
    
    # Retrieve the tool list registered on the MCP server (e.g., sentiment_analysis) as "tools = mcp_client.get_tools()"
    tools = mcp_client.get_tools()
    
    # Use the OpenAI model exposed by the MCP server (e.g., gpt-4o) as "model = OpenAIServerModel(model_id="gpt-4o")"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_id = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    if not model_id:
        raise RuntimeError("Set OPENAI_MODEL before starting the client.")

    model = OpenAIServerModel(model_id=model_id, api_key=api_key)

    # Create a CodeAgent that can reason with tools + code
    agent = CodeAgent(
        # Pass the initialized `tools` as "tools=tools,"
        tools=tools,
        # Pass the initialized `models` as "model=model,"
        model = model,
        additional_authorized_imports=["json", "ast", "urllib", "base64"]
    )

    # Function to handle each incoming user message
    def smart_agent(message, history):
        # Prompt that guides the model to either respond directly or trigger the tool+code execution
        decision_prompt = f"""You are a smart AI assistant.

Your job is to decide how to respond to the user's message.

If it's a vague or open-ended prompt (e.g., "Can you help me?", "What's up?"),
respond naturally in plain text — greet the user or ask for more input.

If the request requires tools, code, or calculations, reply only with:
<use_tools>

When the message asks for sentiment analysis (e.g., "What is the sentiment of ...", "Analyze this text for sentiment", "Do sentiment analysis of ..."), 
ALWAYS use the sentiment_analysis tool available via the MCP interface.

When the message asks for summarize text or paragraph, ALWAYS use the summarize_text tool available via the MCP interface 

When the message asks for translate to french, ALWAYS use the translate_to_french tool available via the MCP interface

For stock-price or stock-quote requests, ALWAYS use the fetch_stock_price tool available via the MCP interface. Never supply a stock price from memory.
Think internally and respond appropriately.
Do NOT mention category labels like (A) or (B).
Only return what the user should see.

User message:
\"\"\"{message}\"\"\"
"""
        # Ask the model how to respond based on the user's message
        # Use the model's `.generate()` method to get a response based on the decision prompt
        # as "llm_reply = model.generate(messages=[ChatMessage(role="user", content=decision_prompt)])"
        llm_reply = model.generate(messages=[ChatMessage(role="user", content=decision_prompt)])
        print("LLM reply:", llm_reply.content)

        reply = llm_reply.content or ""

        needs_tools = any(
            marker in reply
            for marker in (
                "<use_tools>",
                "<summarize_text>",
                "<sentiment_analysis>",
                "<translate_to_french>",
                "<fetch_stock_price>",
            )
        )
        # If the model decides that tool + code execution is needed
        if needs_tools:
            # Trigger the full agentic loop: reasoning + tool execution
            return str(agent.run(
                "Use the available MCP summarize_text tool for summarization "
                "and sentiment_analysis tool for sentiment analysis. "
                "and translate_to_french tool for translation to French. "
                "Use fetch_stock_price for stock quotes, passing the ticker symbol. "
                "If the symbol is unclear, ask the user to specify it. "
                "Report the returned price, symbol"
                "Do not invent prices or assume currency, and do not describe the quote as real-time unless the tool confirms it. "
                "Return the tool's result to the user. "
                f"User request: {message}"
            ))
        else:
            # Otherwise, return the plain model-generated reply
            return llm_reply.content.strip()

    # Gradio UI (unchanged)
    demo = gr.ChatInterface(
        fn=smart_agent, # Function to handle messages
        type="messages", # Accept message-based input
        examples=[
            "Analyze the sentiment of the following text: 'This is awesome'",
            "Fetch the latest available stock price for MSFT.",
        ],
        title="Agent with MCP Tools",  # UI Title
        description="Smart agent: quick replies for simple asks, full reasoning for complex ones.", # UI Description
    )
    # Launch the Gradio app locally
    demo.launch()

finally:
    # Disconnect from the MCP server when the script exits
    mcp_client.disconnect()
