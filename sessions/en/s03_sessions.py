"""
Section 03: Sessions & Context Guard
"Sessions are JSONL files. Append on write, replay on read. When too big, summarize."

Two layers around the same agent loop:

  SessionStore -- JSONL persistence (append on write, replay on read)
  ContextGuard -- 3-stage overflow retry:
    try normal -> truncate tool results -> compact history (50%) -> fail

    User Input
        |
    load_session() --> rebuild messages[] from JSONL
        |
    guard_api_call() --> try -> truncate -> compact -> raise
        |
    save_turn() --> append to JSONL
        |
    Print response

Usage:
    cd claw0
    python en/s03_sessions.py

Required .env:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os          # For reading environment variables
import sys         # For sys.exit() on configuration errors
import json        # For JSONL reading/writing
import uuid        # For generating unique session IDs
import time        # For timestamps
from pathlib import Path  # For path manipulation
from datetime import datetime, timezone  # For timestamps
from typing import Any   # For type hints

# Third-party packages
from dotenv import load_dotenv   # Loads .env file into environment variables
from anthropic import Anthropic  # Anthropic's official SDK for Claude API

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# System prompt - instructs model to use tools and maintain session context
SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

# Workspace directory - all file operations restricted here
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

# Context window safety limit (180k tokens = conservative for 200k window)
CONTEXT_SAFE_LIMIT = 180000

# Maximum characters to return from tool output
MAX_TOOL_OUTPUT = 50000

# ---------------------------------------------------------------------------
# ANSI Colors for terminal output
# ---------------------------------------------------------------------------
CYAN = "\033[36m"      # User prompts
GREEN = "\033[32m"     # Assistant responses
YELLOW = "\033[33m"    # Warnings
RED = "\033[31m"       # Errors
DIM = "\033[2m"        # Info text
RESET = "\033[0m"      # Reset formatting
BOLD = "\033[1m"       # Bold text
MAGENTA = "\033[35m"   # Session info


def colored_prompt() -> str:
    """Returns the styled prompt string shown to user before input."""
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    """Prints the assistant's response with green color scheme."""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    """Prints tool invocation info (dimmed)."""
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    """Prints informational text with dim formatting."""
    print(f"{DIM}{text}{RESET}")


def print_warn(text: str) -> None:
    """Prints warnings with yellow color."""
    print(f"{YELLOW}{text}{RESET}")


def print_session(text: str) -> None:
    """Prints session-related info with magenta color."""
    print(f"{MAGENTA}{text}{RESET}")


# ---------------------------------------------------------------------------
# Safe path helper
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """Resolve a path and ensure it stays within WORKSPACE_DIR.
    
    This is a security function that prevents path traversal attacks.
    It ensures users can't access files outside the workspace directory.
    """
    target = (WORKSPACE_DIR / raw).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target


# ---------------------------------------------------------------------------
# SessionStore -- JSONL-based conversation persistence
# ---------------------------------------------------------------------------
# Mental model: each session is a .jsonl file (JSON Lines - one JSON object per line).
# Every event (user message, assistant response, tool call, tool result) is one JSON line.
#
# To restore a session:
#   1. Read all lines from the JSONL file
#   2. Parse each line as JSON
#   3. Rebuild the API messages array in the correct format
#
# Why JSONL?
#   - Append-only: Easy to add new turns without rewriting the whole file
#   - Line-based: Can read/write without loading entire file into memory
#   - Simple: No database needed, just files on disk
#
# Directory structure:
#   workspace/.sessions/agents/<agent_id>/sessions/
#       sessions.json      # Index of all sessions (metadata)
#       <session_id>.jsonl # Actual conversation data
# ---------------------------------------------------------------------------


class SessionStore:
    """Manages persistent storage for agent sessions using JSONL files.
    
    Each session is stored as a .jsonl file where each line is a JSON object
    representing one turn in the conversation (user message, assistant message,
    tool call, or tool result).
    """

    def __init__(self, agent_id: str = "default"):
        """Initialize the session store for a specific agent.
        
        Args:
            agent_id: Identifier for the agent (default: "default")
                     Creates separate session files per agent
        """
        self.agent_id = agent_id
        # Base directory for all sessions
        self.base_dir = WORKSPACE_DIR / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Index file tracks all sessions and their metadata
        self.index_path = self.base_dir.parent / "sessions.json"
        self._index: dict[str, dict] = self._load_index()
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        """Load the sessions index from the JSON file.
        
        Returns:
            Dictionary mapping session_id to session metadata
        """
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        """Save the sessions index to the JSON file."""
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session's JSONL file."""
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        """Create a new session and return its ID.
        
        Args:
            label: Optional label/description for the session
            
        Returns:
            The new session ID (12-character hex string)
        """
        # Generate a short unique ID (first 12 chars of UUID hex)
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        
        # Store metadata in index
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        
        # Create the empty JSONL file
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict]:
        """Rebuild API-format messages[] from JSONL file.
        
        This is the key function that restores a conversation. It reads
        the JSONL file and reconstructs the messages array that can be
        passed to the Anthropic API.
        
        Args:
            session_id: The session to load
            
        Returns:
            List of message dictionaries in API format
        """
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        """Save a user or assistant message to the current session.
        
        Args:
            role: "user" or "assistant"
            content: The message content (string or list of blocks)
        """
        if not self.current_session_id:
            return
        self.append_transcript(self.current_session_id, {
            "type": role,
            "content": content,
            "ts": time.time(),
        })

    def save_tool_result(self, tool_use_id: str, name: str,
                         tool_input: dict, result: str) -> None:
        """Save both the tool_use and tool_result to the transcript.
        
        When a tool is called, we save two records:
        1. The tool_use (what was called)
        2. The tool_result (what was returned)
        
        Both get the same timestamp so they stay paired.
        """
        if not self.current_session_id:
            return
        ts = time.time()
        
        # Save the tool call
        self.append_transcript(self.current_session_id, {
            "type": "tool_use",
            "tool_use_id": tool_use_id,
            "name": name,
            "input": tool_input,
            "ts": ts,
        })
        # Save the result
        self.append_transcript(self.current_session_id, {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "ts": ts,
        })

    def append_transcript(self, session_id: str, record: dict) -> None:
        """Append a JSON record to the session's JSONL file.
        
        This is the "append on write" part - we just add a new line
        to the file, no need to rewrite anything.
        """
        path = self._session_path(session_id)
        # Open in append mode, write one JSON line
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Update index metadata
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    def _rebuild_history(self, path: Path) -> list[dict]:
        """Rebuild API-format messages from JSONL lines.
        
        This is the "replay on read" part. We transform the stored
        format back to the API message format.
        
        Key transformation rules:
        - "user" type -> {"role": "user", "content": ...}
        - "assistant" type -> {"role": "assistant", "content": ...}
        - "tool_use" blocks go INTO assistant messages (as blocks)
        - "tool_result" blocks go INTO user messages (as blocks)
        
        Messages must alternate user/assistant for the API to work.
        """
        messages: list[dict] = []
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            # User message - simple conversion
            if rtype == "user":
                messages.append({
                    "role": "user",
                    "content": record["content"],
                })

            # Assistant message - could contain text and/or tool_use blocks
            elif rtype == "assistant":
                content = record["content"]
                # Handle legacy format where content was just a string
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            # Tool use block - add to the last assistant message
            elif rtype == "tool_use":
                block = {
                    "type": "tool_use",
                    "id": record["tool_use_id"],
                    "name": record["name"],
                    "input": record["input"],
                }
                # Append to existing assistant message if one exists
                if messages and messages[-1]["role"] == "assistant":
                    content = messages[-1]["content"]
                    if isinstance(content, list):
                        content.append(block)
                    else:
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(content)},
                            block,
                        ]
                else:
                    # Or create new assistant message with just this tool_use
                    messages.append({
                        "role": "assistant",
                        "content": [block],
                    })

            # Tool result block - add to the last user message
            elif rtype == "tool_result":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": record["tool_use_id"],
                    "content": record["content"],
                }
                # Append to existing user message if it's already a tool_result message
                if (messages and messages[-1]["role"] == "user"
                        and isinstance(messages[-1]["content"], list)
                        and messages[-1]["content"]
                        and isinstance(messages[-1]["content"][0], dict)
                        and messages[-1]["content"][0].get("type") == "tool_result"):
                    messages[-1]["content"].append(result_block)
                else:
                    # Or create new user message with this tool_result
                    messages.append({
                        "role": "user",
                        "content": [result_block],
                    })

        return messages

    def list_sessions(self) -> list[tuple[str, dict]]:
        """List all sessions, sorted by last_active (most recent first).
        
        Returns:
            List of (session_id, metadata) tuples
        """
        items = list(self._index.items())
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items


def _serialize_messages_for_summary(messages: list[dict]) -> str:
    """Flatten messages to plain text for LLM summarization.
    
    When we need to compress old messages into a summary, we first
    convert them to a simple text format that the summarizer can understand.
    
    Format:
        [user]: message content
        [assistant]: response
        [assistant called tool_name]: {"arg": "value"}
        [tool_result]: result preview
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block['text']}")
                    elif btype == "tool_use":
                        parts.append(
                            f"[{role} called {block.get('name', '?')}]: "
                            f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                        )
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                        parts.append(f"[tool_result]: {preview}")
                elif hasattr(block, "text"):
                    parts.append(f"[{role}]: {block.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ContextGuard -- context overflow protection
# ---------------------------------------------------------------------------
# The context window is limited (e.g., 200k tokens for Claude 3.5).
# When messages grow too large, the API throws a context overflow error.
# ContextGuard handles this with a 3-stage retry strategy:
#
#   Stage 1 (attempt 0): Try the API call normally
#   Stage 2 (attempt 1): If overflow, truncate oversized tool_result blocks
#                        Keep only the first 30% of each tool's output
#   Stage 3 (attempt 2): If still overflowing, compact history via LLM
#                        Summarize the oldest 50% of messages, keep recent 20%
#   Stage 4: If still failing after all retries, raise the error
#
# This gives the conversation a chance to continue even when it gets long.
# ---------------------------------------------------------------------------


class ContextGuard:
    """Protect the agent from context window overflow with 3-stage retry."""

    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        """Initialize the context guard.
        
        Args:
            max_tokens: Safety limit for estimated token count (default 180k)
        """
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate: 1 token per 4 characters.
        
        This is a very rough heuristic. The actual tokenization depends
        on the model and varies by content. For safety, we underestimate
        (fewer tokens = more conservative).
        """
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens in the messages array.
        
        Counts text content and tool results, but not the overhead
        of the message structure itself.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {}))
                            )
                    elif hasattr(block, "text"):
                        total += self.estimate_tokens(block.text)
                    elif hasattr(block, "input"):
                        total += self.estimate_tokens(json.dumps(block.input))
        return total

    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        """Head-only truncation: keep only the first portion of tool output.
        
        Tool outputs (especially file reads) can be huge. We keep only
        the first max_fraction of the context budget to leave room for
        the actual conversation.

        So it truncates each tool_result to:
        1 token = 4 chars = 3/4 word
        - 180,000 * 4 * 0.3 = 216,000 characters (30% of context budget)
        Only the tool result content gets truncated, not regular messages. 
        It keeps the first 30% of the tool output and discards the rest with a note.
        
        Args:
            result: The tool output string to truncate
            max_fraction: Maximum fraction of context to use (default 30%)
            
        Returns:
            Truncated string with note about total size
        """
        max_chars = int(self.max_tokens * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        head = result[:max_chars]
        return head + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"

    def compact_history(self, messages: list[dict],
                        api_client: Anthropic, model: str) -> list[dict]:
        """Compress old messages into an LLM-generated summary.
        
        Strategy:
        - Keep the most recent 20% of messages (min 4) as-is
        - Summarize the oldest 50% of messages using an LLM call
        - Discard the middle portion
        
        The summary replaces the old messages, dramatically reducing
        token count while preserving key facts and decisions.
        
        Args:
            messages: The current messages array
            api_client: Anthropic client for the summarization call
            model: Model to use for summarization
            
        Returns:
            New messages array with old messages replaced by summary
        """
        total = len(messages)
        if total <= 4:
            return messages

        # Calculate how many to keep vs compress
        #Example with 20 messages:
        #- Compress: oldest 10 messages → summary
        #- Keep: recent 4 messages
        #- Discard: middle 6 messages
        keep_count = max(4, int(total * 0.2))  # Keep at least 4
        compress_count = max(2, int(total * 0.5))  # Compress up to 50%
        compress_count = min(compress_count, total - keep_count)  # Don't exceed available

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]

        # Serialize old messages for summarization
        old_text = _serialize_messages_for_summary(old_messages)

        # Prompt for the summarizer
        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            # Call LLM to generate summary
            summary_resp = api_client.messages.create(
                model=model,
                max_tokens=2048,
                system="You are a conversation summarizer. Be concise and factual.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_text = ""
            for block in summary_resp.content:
                if hasattr(block, "text"):
                    summary_text += block.text

            print_session(
                f"  [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            # If summarization fails, just drop the old messages
            print_warn(f"  [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        # Build the compacted messages
        compacted = [
            # User message with the summary
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            # Assistant acknowledgment
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood, I have the context from our previous conversation."}],
            },
        ]
        # Add the recent messages we kept
        compacted.extend(recent_messages)
        return compacted

    def _truncate_large_tool_results(self, messages: list[dict]) -> list[dict]:
        """Walk through messages and truncate oversized tool_result blocks.
        
        This is Stage 2 of the overflow protection. We scan all messages
        and truncate any tool_result content that exceeds our limit.
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    # Only truncate tool_result blocks
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("content"), str)):
                        block = dict(block)  # Make a copy
                        block["content"] = self.truncate_tool_result(
                            block["content"]
                        )
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    def guard_api_call(
        self,
        api_client: Anthropic,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 2,
    ) -> Any:
        """3-stage retry for API calls with context overflow protection.
        
        Flow:
          Attempt 0: Try the call normally
          Attempt 1: If overflow, truncate tool results and retry
          Attempt 2: If still overflowing, compact history and retry
          If still failing: raise the error
        
        Args:
            api_client: Anthropic client
            model: Model ID to use
            system: System prompt
            messages: Message history
            tools: Optional tools list
            max_retries: Maximum retry attempts (default 2)
            
        Returns:
            The API response
            
        Raises:
            Exception: If all retry strategies fail
        """
        current_messages = messages

        for attempt in range(max_retries + 1):
            try:
                # Build the API call kwargs
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 8096,
                    "system": system,
                    "messages": current_messages,
                }
                if tools:
                    kwargs["tools"] = tools
                
                # Make the API call
                result = api_client.messages.create(**kwargs)
                
                # If we modified messages, update the original
                if current_messages is not messages:
                    messages.clear()
                    messages.extend(current_messages)
                return result

            except Exception as exc:
                # Check if this is a context overflow error
                error_str = str(exc).lower()
                is_overflow = ("context" in error_str or "token" in error_str)

                # If not overflow, or we've exhausted retries, re-raise
                if not is_overflow or attempt >= max_retries:
                    raise

                # Stage 1: Truncate tool results
                if attempt == 0:
                    print_warn(
                        "  [guard] Context overflow, truncating tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(
                        current_messages
                    )
                # Stage 2: Compact history
                elif attempt == 1:
                    print_warn(
                        "  [guard] Still overflowing, compacting history..."
                    )
                    current_messages = self.compact_history(
                        current_messages, api_client, model
                    )

        raise RuntimeError("guard_api_call: exhausted retries")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
# Same as s02, but limited to workspace directory
# ---------------------------------------------------------------------------


def tool_read_file(file_path: str) -> str:
    """Read the contents of a file under the workspace directory."""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_list_directory(directory: str = ".") -> str:
    """List files and subdirectories in a directory under workspace."""
    print_tool("list_directory", directory)
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            prefix = "[dir]  " if entry.is_dir() else "[file] "
            lines.append(prefix + entry.name)
        return "\n".join(lines) if lines else "[empty directory]"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_get_current_time() -> str:
    """Get the current date and time in UTC."""
    print_tool("get_current_time", "")
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Tool schema + dispatch table
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file under the workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to workspace directory.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory under workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path relative to workspace directory. Default is root.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in UTC.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": tool_read_file,
    "list_directory": tool_list_directory,
    "get_current_time": tool_get_current_time,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Look up handler by name, call it with the input kwargs."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------
# Session-aware commands (prefixed with /)
# These allow users to manage sessions without being in the agent loop
# ---------------------------------------------------------------------------

def handle_repl_command(
    command: str,
    store: SessionStore,
    guard: ContextGuard,
    messages: list[dict],
) -> tuple[bool, list[dict]]:
    """Handle /-prefixed commands. Returns (handled, messages).
    
    Commands:
        /new [label]    - Create a new session
        /list           - List all sessions
        /switch <id>   - Switch to a session (prefix match)
        /context        - Show context token usage
        /compact        - Manually compact conversation history
        /help           - Show help
    
    Args:
        command: The full command string
        store: Session store instance
        guard: Context guard instance
        messages: Current message list
        
    Returns:
        Tuple of (command_was_handled, possibly_updated_messages)
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    # /new - Create a new session
    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"  Created new session: {sid}" + (f" ({label})" if label else ""))
        return True, []  # Return empty messages for new session

    # /list - Show all sessions
    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("  No sessions found.")
            return True, messages
        print_info("  Sessions:")
        for sid, meta in sessions:
            active = " <-- current" if sid == store.current_session_id else ""
            label = meta.get("label", "")
            label_str = f" ({label})" if label else ""
            count = meta.get("message_count", 0)
            last = meta.get("last_active", "?")[:19]
            print_info(f"    {sid}{label_str}  msgs={count}  last={last}{active}")
        return True, messages

    # /switch - Switch to a different session
    elif cmd == "/switch":
        if not arg:
            print_warn("  Usage: /switch <session_id>")
            return True, messages
        target_id = arg.strip()
        # Prefix match - user can type just the first few characters
        matched = [sid for sid in store._index if sid.startswith(target_id)]
        if len(matched) == 0:
            print_warn(f"  Session not found: {target_id}")
            return True, messages
        if len(matched) > 1:
            print_warn(f"  Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages
        sid = matched[0]
        new_messages = store.load_session(sid)
        print_session(f"  Switched to session: {sid} ({len(new_messages)} messages)")
        return True, new_messages  # Return loaded messages

    # /context - Show token usage
    elif cmd == "/context":
        estimated = guard.estimate_messages_tokens(messages)
        pct = (estimated / guard.max_tokens) * 100
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        color = GREEN if pct < 50 else (YELLOW if pct < 80 else RED)
        print_info(f"  Context usage: ~{estimated:,} / {guard.max_tokens:,} tokens")
        print(f"  {color}[{bar}] {pct:.1f}%{RESET}")
        print_info(f"  Messages: {len(messages)}")
        return True, messages

    # /compact - Manually trigger history compaction
    elif cmd == "/compact":
        if len(messages) <= 4:
            print_info("  Too few messages to compact (need > 4).")
            return True, messages
        print_session("  Compacting history...")
        new_messages = guard.compact_history(messages, client, MODEL_ID)
        print_session(f"  {len(messages)} -> {len(new_messages)} messages")
        return True, new_messages  # Return compacted messages

    # /help - Show available commands
    elif cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages

    # Not a recognized command - let the agent handle it
    return False, messages


# ---------------------------------------------------------------------------
# Core: Agent loop (same while True, wrapped with SessionStore + ContextGuard)
# ---------------------------------------------------------------------------
# This section adds two new features to the basic agent loop:
#
#   1. Session persistence - conversations survive restarts
#      - On startup: load most recent session or create new one
#      - After each turn: save to JSONL file
#
#   2. Context overflow protection - prevents crashes on long conversations
#      - Use guard.guard_api_call() instead of direct client call
#      - Automatically handles truncation and compaction
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop with session persistence and context protection.
    
    This is the same structure as s02, but with:
    - SessionStore wrapping for persistence
    - ContextGuard wrapping for overflow protection
    - REPL commands for session management (/new, /list, /switch, etc.)
    """

    # Initialize session store and context guard
    store = SessionStore(agent_id="claw0")
    guard = ContextGuard()

    # Resume most recent session or create a new one
    sessions = store.list_sessions()
    if sessions:
        # Load the most recent session
        sid = sessions[0][0]
        messages = store.load_session(sid)
        print_session(f"  Resumed session: {sid} ({len(messages)} messages)")
    else:
        # Create initial session
        sid = store.create_session("initial")
        messages = []
        print_session(f"  Created initial session: {sid}")

    # Print welcome banner
    print_info("=" * 60)
    print_info("  claw0  |  Section 03: Sessions & Context Guard")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Session: {store.current_session_id}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  Type /help for commands, quit/exit to leave.")
    print_info("=" * 60)
    print()

    # Main loop
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
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        # --- Step 2: Handle REPL commands ---
        # Commands start with / and are handled before the agent
        if user_input.startswith("/"):
            handled, messages = handle_repl_command(
                user_input, store, guard, messages
            )
            if handled:
                continue  # Command was handled, skip to next input

        # --- Step 3: Add user message to history and save ---
        messages.append({
            "role": "user",
            "content": user_input,
        })
        store.save_turn("user", user_input)

        # --- Inner loop: Handle tool call chains (same as s02) ---
        while True:
            # Use guard.guard_api_call() instead of direct client call
            # This adds overflow protection
            try:
                response = guard.guard_api_call(
                    api_client=client,
                    model=MODEL_ID,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                # Clean up failed messages
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            # Add assistant response to history and save
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Serialize content blocks for storage
            # API returns objects with attributes, but JSONL needs plain dicts
            # Convert: block.text -> {"type": "text", "text": "..."}
            #          block (tool_use) -> {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
            serialized_content = []
            for block in response.content:
                # Text block: model returned regular text response
                if hasattr(block, "text"):
                    serialized_content.append({"type": "text", "text": block.text})
                # Tool use block: model wants to call a function
                elif block.type == "tool_use":
                    serialized_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            store.save_turn("assistant", serialized_content)

            # Check stop_reason (same as s02)
            if response.stop_reason == "end_turn":
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                break

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = process_tool_call(block.name, block.input)
                    # Save tool result to session
                    store.save_tool_result(
                        block.id, block.name, block.input, result
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue

            else:
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