"""
Section 02: Tool Use
"Tools are data (schema dict) + a handler map. Model picks a name, you look it up."

The agent loop is unchanged from s01. The only additions:
  1. TOOLS array tells the model what tools exist (JSON schema)
  2. TOOL_HANDLERS dict maps tool names to Python functions
  3. When stop_reason == "tool_use", dispatch and feed result back

    User --> LLM --> stop_reason == "tool_use"?
                          |
                  TOOL_HANDLERS[name](**input)
                          |
                  tool_result --> back to LLM
                          |
                   stop_reason == "end_turn"? --> Print

Tools:
    - bash        : Run shell commands
    - read_file   : Read file contents
    - write_file  : Write to a file
    - edit_file   : Exact string replacement in a file

Usage:
    cd claw0
    python en/s02_tool_use.py

Required .env config:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os          # For reading environment variables
import sys         # For sys.exit() on configuration errors
import subprocess  # For running shell commands (tool_bash)
from pathlib import Path  # For path manipulation and file operations
from typing import Any   # For type hints (TOOL_HANDLERS dict)

# Third-party packages (install via pip install python-dotenv anthropic)
from dotenv import load_dotenv   # Loads .env file into environment variables
from anthropic import Anthropic  # Anthropic's official SDK for Claude API

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load the .env file from the project root directory
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# Get model ID from environment, with a sensible default if not set
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

# Create the Anthropic client for API communication
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# System prompt - instructs the model how to use tools
# This tells the model:
#   - It has access to tools for file operations and shell commands
#   - Must read a file before editing it
#   - Must match old_string exactly when using edit_file (no whitespace variations)
SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

# Maximum characters to return from tool output (prevents huge responses)
MAX_TOOL_OUTPUT = 50000
# Working directory - all file operations are restricted to this directory
WORKDIR = Path.cwd()

# ---------------------------------------------------------------------------
# ANSI Colors for terminal output
# ---------------------------------------------------------------------------
CYAN = "\033[36m"      # User prompts
GREEN = "\033[32m"     # Assistant responses
YELLOW = "\033[33m"    # Warnings
RED = "\033[31m"       # Errors
DIM = "\033[2m"        # Info text (muted)
RESET = "\033[0m"      # Reset formatting
BOLD = "\033[1m"       # Bold text


def colored_prompt() -> str:
    """Returns the styled prompt string shown to user before input."""
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    """Prints the assistant's response with green color scheme."""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    """Prints tool invocation info (dimmed, for debugging/tracing)."""
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    """Prints informational text with dim formatting."""
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """Resolve a path and ensure it stays within WORKDIR.
    
    This is a critical security function that prevents path traversal attacks.
    It ensures users (and the AI) can't access files outside the working directory.
    
    Args:
        raw: The path string from user input
        
    Returns:
        Resolved Path object if within WORKDIR
        
    Raises:
        ValueError: If the path resolves outside WORKDIR
    """
    # Resolve to absolute path and check it's within WORKDIR
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Truncate long text to prevent API issues.
    
    Large file reads can produce huge outputs that exceed API limits.
    This truncates and adds a note about total size.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
# Each tool function takes specific arguments and returns a string result.
# The result is always a string (success message or error message).
# This string gets sent back to the model as a "tool_result".
# ---------------------------------------------------------------------------

def tool_bash(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output.
    
    This is the most powerful tool - it can run any shell command.
    We add safety checks to prevent dangerous commands.
    
    Args:
        command: The shell command to execute
        timeout: Maximum seconds to wait (default 30)
        
    Returns:
        Command output (stdout + stderr) or error message
    """
    # Security: Block obviously dangerous commands
    # These patterns could cause irreversible damage
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

    print_tool("bash", command)
    try:
        # subprocess.run executes the command and captures output
        # shell=True allows pipes, redirects, etc.
        # capture_output=True captures both stdout and stderr
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKDIR),  # Run in our working directory
        )
        output = ""
        # Combine stdout and stderr, mark stderr clearly
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        # Include exit code if non-zero (indicates error)
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file (relative to working directory)
        
    Returns:
        File contents or error message
    """
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)  # Security check
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)  # Truncate if too large
    except ValueError as exc:
        return str(exc)  # Path traversal error
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.
    
    Args:
        file_path: Path to the file (relative to working directory)
        content: The text content to write
        
    Returns:
        Success message or error message
    """
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path)
        # Create parent directories if they don't exist
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Exact string replacement. old_string must appear exactly once.
    
    This is a precise find-and-replace. The model must:
    1. First read the file to see the exact text
    2. Provide that exact text as old_string
    3. Provide the new text as new_string
    
    Args:
        file_path: Path to the file
        old_string: Exact text to find and replace
        new_string: Replacement text
        
    Returns:
        Success message or error message
    """
    print_tool("edit_file", f"{file_path} (replace {len(old_string)} chars)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)

        # Validate: old_string must exist exactly once
        # Zero matches = user provided wrong text
        # Multiple matches = ambiguous, need more context
        if count == 0:
            return "Error: old_string not found in file. Make sure it matches exactly."
        if count > 1:
            return (
                f"Error: old_string found {count} times. "
                "It must be unique. Provide more surrounding context."
            )

        # Perform the replacement (replace only first occurrence)
        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tool schema + dispatch table
# ---------------------------------------------------------------------------
# These are the two key components that make tool calling work:
#
# TOOLS: List of tool definitions (JSON schema format)
#        - This is what the MODEL sees to decide which tool to call
#        - Must include: name, description, input_schema
#        - The description is crucial - model reads it to understand when to use the tool
#
# TOOL_HANDLERS: Dict mapping tool names to Python functions
#        - This is what our CODE uses to actually execute tools
#        - Key = tool name from TOOLS, Value = Python function
# ---------------------------------------------------------------------------

# Tool definitions - passed to the API so the model knows what's available
# This is the ONLY thing the model uses to decide which tool to call
TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with a new string. "
            "The old_string must appear exactly once in the file. "
            "Always read the file first to get the exact text to replace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]

# Handler map - our code uses this to execute tools
# Maps tool name (from TOOLS) -> Python function
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Look up handler by name, call it with the input kwargs.
    
    This is the bridge between the model's request and our Python code.
    1. Look up the tool name in TOOL_HANDLERS
    2. Call the handler function with the model's provided arguments
    3. Return the result as a string (to be sent back to the model)
    
    Args:
        tool_name: Name of the tool to call (from model)
        tool_input: Dictionary of arguments (from model)
        
    Returns:
        Tool execution result (string) or error message
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        # Call the handler function with **kwargs (unpack dict as named args)
        # Example: tool_bash(command="ls", timeout=30)
        return handler(**tool_input)
    except TypeError as exc:
        # Wrong arguments provided
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        # Tool execution failed
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# Core: The Agent Loop (same while True as s01, plus tool dispatch)
# ---------------------------------------------------------------------------
# Key difference from s01: The INNER while loop handles tool use
#
# Flow:
#   1. User enters message
#   2. Call API with tools=TOOLS (model sees available tools)
#   3. If stop_reason == "end_turn": Print response, break to outer loop
#   4. If stop_reason == "tool_use": 
#        - Execute each tool call
#        - Append results as a user message
#        - Continue inner loop (call API again with results)
#   5. Model may chain multiple tool calls before giving final answer
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop - REPL with tool support.
    
    This is the same structure as s01, but with an inner loop
    to handle the tool_use -> execute -> result -> continue cycle.
    """

    # messages stores the full conversation history
    messages: list[dict] = []

    # Print welcome banner
    print_info("=" * 60)
    print_info("  claw0  |  Section 02: Tool Use")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  Type 'quit' or 'exit' to leave. Ctrl+C also works.")
    print_info("=" * 60)
    print()

    # Outer loop - each iteration is one user message
    while True:
        # --- Step 1: Read user input ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        # --- Step 2: Add user message to history ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- Inner loop: Handle tool calls (model may call multiple) ---
        # This is the key difference from s01: we loop until end_turn
        # Each iteration either:
        #   - Gets tool calls, executes them, continues
        #   - Gets final text response, prints it, exits
        while True:
            try:
                # Call API with tools parameter - model sees available tools
                # If user request needs a tool, model will request it
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=8096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,  # NEW: Tell model what tools exist
                    messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                # Clean up failed message from history
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            # Add assistant response to history (may contain tool_use blocks)
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # --- Step 3: Check stop_reason ---
            if response.stop_reason == "end_turn":
                # Model gave a final text response (possibly after using tools)
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                break  # Exit inner loop, back to outer loop for next user input

            elif response.stop_reason == "tool_use":
                # Model wants to use one or more tools
                # We must execute each tool and return results
                tool_results = []
                
                # Iterate through response content blocks
                for block in response.content:
                    # Skip any text blocks, only process tool_use blocks
                    if block.type != "tool_use":
                        continue
                    
                    # Execute the tool: block.name = tool name, block.input = arguments
                    result = process_tool_call(block.name, block.input)
                    
                    # Build tool_result in Anthropic's expected format
                    # type: "tool_result"
                    # tool_use_id: Links result to the specific tool_use block
                    # content: The result string from tool execution
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                # Append tool results as a user message
                # IMPORTANT: Tool results MUST be in a user message (Anthropic API requirement)
                # This continues the conversation so model can use the results
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                
                # Continue inner loop - call API again with tool results
                # Model will now see the tool outputs and can provide final answer
                continue

            else:
                # Handle other stop_reasons (max_tokens, stop_sequence, etc.)
                print_info(f"[stop_reason={response.stop_reason}]")
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point - validates config before starting the loop."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()