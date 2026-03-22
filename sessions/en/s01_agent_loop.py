"""
Section 01: The Agent Loop
"An agent is just while True + stop_reason"

    User Input --> [messages[]] --> LLM API --> stop_reason?
                                                /        \
                                          "end_turn"  "tool_use"
                                              |           |
                                           Print      (next section)

Usage:
    cd claw0
    python en/s01_agent_loop.py

Required .env config:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Standard library modules
import os          # For reading environment variables
import sys         # For sys.exit() to handle errors
from pathlib import Path  # For constructing paths to .env file

# Third-party packages (install via pip install python-dotenv anthropic)
from dotenv import load_dotenv   # Loads .env file into environment variables
from anthropic import Anthropic  # Anthropic's official SDK for Claude API

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load the .env file from the project root directory
# Path(__file__).resolve().parent.parent.parent gives us the project root
# override=True ensures we overwrite any existing env vars with .env values
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# Get model ID from environment, with a sensible default if not set
# This determines which Claude model we use (e.g., claude-sonnet-4-20250514)
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

# Create the Anthropic client - this handles all API communication
# api_key: Your Anthropic API key (set in .env)
# base_url: Optional - use for compatible providers like OpenRouter
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# System prompt sets the AI's behavior and personality
# This runs before every conversation and shapes how the model responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI Colors for terminal output formatting
# ---------------------------------------------------------------------------
# These escape codes make the CLI output more readable and visually appealing
# \033[XXm is the ANSI escape sequence format
CYAN = "\033[36m"      # For user prompts
GREEN = "\033[32m"     # For assistant responses
YELLOW = "\033[33m"    # For warnings/errors
DIM = "\033[2m"        # For info/dim text (muted)
RESET = "\033[0m"      # Reset all formatting
BOLD = "\033[1m"       # Bold text


def colored_prompt() -> str:
    """Returns the styled prompt string shown to user before input."""
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    """Prints the assistant's response with green color scheme."""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    """Prints informational text with dim formatting."""
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# Core: The Agent Loop
# ---------------------------------------------------------------------------
# The agent loop is the heart of any AI agent system. It follows this pattern:
#
# 1. Collect user input, append to messages list
# 2. Call the LLM API with the full message history
# 3. Check the stop_reason in the response:
#    - "end_turn": The model finished its response, print it to user
#    - "tool_use": The model wants to use a tool (handled in next section)
# 4. Append the assistant's response to message history
# 5. Repeat
#
# Key insight: The loop structure stays the same whether or not tools exist.
# We just handle different stop_reasons. This is the fundamental pattern
# that all agent systems follow.
#
# In this section, stop_reason is always "end_turn" because we haven't added
# tool support yet. The next section (s02_tool_use.py) will add tools, but
# the loop structure remains identical.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop - implements a conversational REPL.
    
    REPL = Read-Eval-Print Loop. This creates an interactive terminal
    session where the user types messages and gets AI responses.
    
    The loop maintains a message history (context) so the AI can
    reference previous exchanges in the conversation.
    """

    # messages stores the full conversation history
    # Each message is a dict with "role" (user/assistant) and "content"
    # We pass this entire history to the API on each turn
    messages: list[dict] = []

    # Print welcome banner
    print_info("=" * 60)
    print_info("  claw0  |  Section 01: The Agent Loop")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  Type 'quit' or 'exit' to leave. Ctrl+C also works.")
    print_info("=" * 60)
    print()

    # Main loop - runs forever until user quits
    while True:
        # --- Step 1: Read user input ---
        try:
            # input() blocks until user types something and presses Enter
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C gracefully
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Check for quit commands
        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        # --- Step 2: Add user message to history ---
        # We append BEFORE calling the API so the model sees the new message
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- Step 3: Call the LLM API ---
        try:
            # client.messages.create() is the main Anthropic API call
            # Parameters:
            #   model: Which Claude model to use
            #   max_tokens: Max response length (prevents infinite outputs)
            #   system: Sets the AI's behavior/instructions
            #   messages: Full conversation history
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            # Handle API errors gracefully - show error but don't crash
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            # Remove the user message we just added since it failed
            messages.pop()
            continue

        # --- Step 4: Process the response based on stop_reason ---
        # stop_reason tells us WHY the model stopped generating
        # This is the critical piece that makes the agent loop work
        
        if response.stop_reason == "end_turn":
            # Model finished generating text normally
            # Extract text content from response blocks
            # (Response can contain multiple content blocks)
            assistant_text = ""
            for block in response.content:
                # Each block has different types - we extract text blocks
                if hasattr(block, "text"):
                    assistant_text += block.text

            # Print the assistant's response
            print_assistant(assistant_text)

            # Add assistant's response to message history
            # This maintains context for the next turn
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        elif response.stop_reason == "tool_use":
            # Model wants to use a tool (function call)
            # This section doesn't have tools yet, so we just inform the user
            print_info("[stop_reason=tool_use] No tools in this section.")
            print_info("See s02_tool_use.py for tool support.")
            
            # Still add the response to history
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        else:
            # Handle other stop_reasons (max_tokens, etc.)
            print_info(f"[stop_reason={response.stop_reason}]")
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                print_assistant(assistant_text)
            
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        # Loop continues - go back to Step 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point - validates config before starting the loop."""
    
    # Check that API key is set - refuse to run without it
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)  # Exit with error code 1

    # Start the agent loop
    agent_loop()


if __name__ == "__main__":
    # This pattern allows the file to be imported as a module
    # or run directly as a script
    main()