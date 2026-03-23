"""
Section 04: Channels -- "Same brain, many mouths"

A Channel encapsulates platform differences so the agent loop only sees
a unified InboundMessage. Adding a new platform = implement receive() +
send(); the loop stays unchanged.

    Telegram ----.                          .---- sendMessage API
    Feishu -------+-- InboundMessage ---+---- im/v1/messages
    CLI (stdin) --'    Agent Loop        '---- print(stdout)

Mental model: every platform is different, but they all produce the same
InboundMessage. One interface, N implementations.

How to run:  cd claw0 && python en/s04_channels.py

Required in .env:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
    # Optional: TELEGRAM_BOT_TOKEN, FEISHU_APP_ID, FEISHU_APP_SECRET
"""

# Standard library imports for JSON, OS, system, time, and threading operations
import json, os, sys, time, threading
from abc import ABC, abstractmethod  # Abstract Base Class for defining interfaces
from dataclasses import dataclass, field  # Data classes for structured objects
from pathlib import Path  # Path manipulation for file operations
from typing import Any  # Type hints for flexible arguments

from dotenv import load_dotenv  # Load environment variables from .env file
from anthropic import Anthropic  # Anthropic Claude API client

# Optional HTTP client - required for Telegram and Feishu channels
# Falls back gracefully if not installed
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env file in project root
# override=True ensures we reload if .env changed during runtime
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# Claude model to use - defaults to Sonnet 4 if not specified
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

# Anthropic API client - communicates with Claude
# Uses ANTHROPIC_BASE_URL if set (useful for proxy/custom endpoints)
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# Workspace directory for memory files and other runtime data
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# State directory for persistent data like Telegram offset tracking
STATE_DIR = WORKSPACE_DIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# System prompt defines the AI's persona and capabilities
SYSTEM_PROMPT = (
    "You are a helpful AI assistant connected to multiple messaging channels.\n"
    "You can save and search notes using the provided tools.\n"
    "When responding, be concise and helpful."
)

# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------
# Color codes for readable console output
CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
BOLD, RED, BLUE = "\033[1m", "\033[31m", "\033[34m"


def print_assistant(text: str, ch: str = "cli") -> None:
    """Print assistant response with channel prefix for non-CLI channels."""
    prefix = f"[{ch}] " if ch != "cli" else ""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {prefix}{text}\n")

def print_tool(name: str, detail: str) -> None:
    """Print tool call information in dim style."""
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")

def print_info(text: str) -> None:
    """Print general information in dim style."""
    print(f"{DIM}{text}{RESET}")

def print_channel(text: str) -> None:
    """Print channel-related messages in blue."""
    print(f"{BLUE}{text}{RESET}")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InboundMessage:
    """
    Unified message format that all channels convert to.
    The agent loop only sees this - platform-specific details are abstracted away.
    
    Fields:
        text: The message content
        sender_id: Unique identifier of the sender (user ID in the platform)
        channel: Channel name (cli, telegram, feishu)
        account_id: Which bot account received this message
        peer_id: The conversation/chat identifier (chat ID for groups, user ID for private)
        is_group: Whether this is a group conversation
        media: List of media attachments (images, files, etc.)
        raw: Raw platform-specific data for debugging
    """
    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

@dataclass
class ChannelAccount:
    """
    Configuration for a single bot instance on a channel.
    One channel type can run multiple bots (e.g., multiple Telegram bots).
    
    Fields:
        channel: Channel type (telegram, feishu, cli)
        account_id: Unique identifier for this bot account
        token: API token for authentication
        config: Channel-specific configuration (allowed chats, app IDs, etc.)
    """
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Session key - unique identifier for a conversation thread
# ---------------------------------------------------------------------------

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    """
    Create a unique key for tracking conversation history.
    Format: agent:main:direct:{channel}:{peer_id}
    
    This key differentiates between:
    - Different channels (telegram vs feishu)
    - Different bot accounts on the same channel
    - Different users/chats within the same account
    """
    return f"agent:main:direct:{channel}:{peer_id}"

# ---------------------------------------------------------------------------
# Channel ABC - Abstract Base Class for all channel implementations
# ---------------------------------------------------------------------------

class Channel(ABC):
    """
    Abstract base class defining the interface for all channel implementations.
    
    To add a new messaging platform, implement:
    - receive(): Poll for new messages (or handle webhooks)
    - send(): Send a message back to the user
    
    The agent loop works with any Channel subclass - it doesn't know or care
    which platform is being used.
    """
    name: str = "unknown"  # Channel identifier (cli, telegram, feishu, etc.)

    @abstractmethod
    def receive(self) -> InboundMessage | None:
        """
        Poll for new incoming messages.
        
        Returns:
            InboundMessage if a message is available, None otherwise.
            For webhook-based channels (Feishu), this returns None as messages
            come via parse_event() instead.
        """
        ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        """
        Send a message to a user or chat.
        
        Args:
            to: The destination (chat ID, user ID, etc.)
            text: Message content to send
            **kwargs: Additional platform-specific options
            
        Returns:
            True if send succeeded, False otherwise.
        """
        ...

    def close(self) -> None:
        """
        Clean up resources (close HTTP connections, file handles, etc.).
        Called when shutting down the agent.
        """
        pass

# ---------------------------------------------------------------------------
# CLIChannel - Command Line Interface channel for local testing
# ---------------------------------------------------------------------------

class CLIChannel(Channel):
    """
    Simple stdin/stdout channel for local testing.
    No API calls, just reads from terminal input and prints responses.
    """
    name = "cli"

    def __init__(self) -> None:
        self.account_id = "cli-local"

    def receive(self) -> InboundMessage | None:
        """
        Read a line from stdin.
        
        Returns:
            InboundMessage with the user's input, or None on Ctrl+C/EOF.
        """
        try:
            text = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        """Print the assistant's response to stdout."""
        print_assistant(text)
        return True

# ---------------------------------------------------------------------------
# Offset persistence -- for tracking Telegram message position
# ---------------------------------------------------------------------------

def save_offset(path: Path, offset: int) -> None:
    """
    Save the Telegram update offset to disk.
    This allows resuming polling from where we left off after restart.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(offset))

def load_offset(path: Path) -> int:
    """
    Load the last known Telegram update offset.
    Returns 0 if no offset file exists or it's corrupted.
    """
    try:
        return int(path.read_text().strip())
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# TelegramChannel -- Bot API long-polling
# ---------------------------------------------------------------------------

class TelegramChannel(Channel):
    """
    Telegram Bot API channel using long-polling.
    
    Features:
    - Long-polling with persistent offset for reliability
    - Media group buffering (combines multiple photos/videos into one message)
    - Text coalescing (reassembles long pastes that Telegram splits)
    - Forum topic support (for supergroups with topics)
    - Allowed chats filtering (security - only respond to specific chats)
    
    Note: Requires httpx package and TELEGRAM_BOT_TOKEN in .env
    """
    name = "telegram"
    MAX_MSG_LEN = 4096  # Telegram's message length limit

    def __init__(self, account: ChannelAccount) -> None:
        """Initialize Telegram channel with bot token."""
        if not HAS_HTTPX:
            raise RuntimeError("TelegramChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        # Telegram Bot API base URL
        self.base_url = f"https://api.telegram.org/bot{account.token}"
        self._http = httpx.Client(timeout=35.0)
        
        # Security: restrict to specific chats (comma-separated chat IDs)
        raw = account.config.get("allowed_chats", "")
        self.allowed_chats = {c.strip() for c in raw.split(",") if c.strip()} if raw else set()

        # Persistence: track position in update stream across restarts
        self._offset_path = STATE_DIR / "telegram" / f"offset-{self.account_id}.txt"
        self._offset = load_offset(self._offset_path)

        # Simple dedup: set of seen update IDs, cleared periodically to bound memory
        self._seen: set[int] = set()

        # Media group buffer: group_id -> {ts, entries}
        # Groups multiple photos/videos sent together into one message
        self._media_buf: dict[str, dict] = {}

        # Text coalesce buffer: (peer, sender) -> {text, msg, ts}
        # Reassembles long pastes that Telegram sends as multiple messages
        self._text_buf: dict[tuple[str, str], dict] = {}

    def _api(self, method: str, **params: Any) -> dict:
        """
        Make a call to the Telegram Bot API.
        
        Args:
            method: API method name (e.g., "sendMessage", "getUpdates")
            **params: Parameters for the API method
            
        Returns:
            API response data, or empty dict on error
        """
        # Filter out None values - Telegram API doesn't accept them
        filtered = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._http.post(f"{self.base_url}/{method}", json=filtered)
            data = resp.json()
            if not data.get("ok"):
                print(f"  {RED}[telegram] {method}: {data.get('description', '?')}{RESET}")
                return {}
            return data.get("result", {})
        except Exception as exc:
            print(f"  {RED}[telegram] {method}: {exc}{RESET}")
            return {}

    def send_typing(self, chat_id: str) -> None:
        """Show "typing..." indicator in the chat."""
        self._api("sendChatAction", chat_id=chat_id, action="typing")

    # -- Polling --

    def poll(self) -> list[InboundMessage]:
        """
        Poll for new messages from Telegram.
        
        Uses long-polling (30s timeout) with offset tracking.
        Returns buffered messages (media groups, coalesced text) that are ready.
        """
        result = self._api("getUpdates", offset=self._offset, timeout=30,
                           allowed_updates=["message"])
        if not result or not isinstance(result, list):
            return self._flush_all()

        for update in result:
            uid = update.get("update_id", 0)

            # Advance offset so Telegram won't re-send these updates on next poll
            if uid >= self._offset:
                self._offset = uid + 1
                save_offset(self._offset_path, self._offset)

            # Simple dedup via set; clear at 5000 to bound memory
            if uid in self._seen:
                continue
            self._seen.add(uid)
            if len(self._seen) > 5000:
                self._seen.clear()

            msg = update.get("message")
            if not msg:
                continue

            # Media groups get buffered separately (multiple updates = one album)
            # Telegram sends each media in a group as separate updates
            if msg.get("media_group_id"):
                mgid = msg["media_group_id"]
                if mgid not in self._media_buf:
                    self._media_buf[mgid] = {"ts": time.monotonic(), "entries": []}
                self._media_buf[mgid]["entries"].append((msg, update))
                continue

            inbound = self._parse(msg, update)
            if not inbound:
                continue
            # Apply allowed chats filter (security)
            if self.allowed_chats and inbound.peer_id not in self.allowed_chats:
                continue

            # Buffer text for coalescing (Telegram splits long pastes into multiple messages)
            key = (inbound.peer_id, inbound.sender_id)
            now = time.monotonic()
            if key in self._text_buf:
                self._text_buf[key]["text"] += "\n" + inbound.text
                self._text_buf[key]["ts"] = now
            else:
                self._text_buf[key] = {"text": inbound.text, "msg": inbound, "ts": now}

        return self._flush_all()

    # -- Flush buffered messages --

    def _flush_all(self) -> list[InboundMessage]:
        """
        Flush expired buffers and return ready messages.
        
        - Media groups: release after 500ms of no new additions
        - Text coalescing: release after 1s of no new additions
        """
        ready: list[InboundMessage] = []

        # Flush media groups after 500ms silence
        now = time.monotonic()
        expired_mg = [k for k, g in self._media_buf.items() if (now - g["ts"]) >= 0.5]
        for mgid in expired_mg:
            entries = self._media_buf.pop(mgid)["entries"]
            captions, media_items = [], []
            for m, _ in entries:
                if m.get("caption"):
                    captions.append(m["caption"])
                # Extract file IDs from various media types
                for mt in ("photo", "video", "document", "audio"):
                    if mt in m:
                        raw_m = m[mt]
                        if isinstance(raw_m, list) and raw_m:
                            fid = raw_m[-1]["file_id"]
                        elif isinstance(raw_m, dict):
                            fid = raw_m.get("file_id", "")
                        else:
                            fid = ""
                        media_items.append({"type": mt, "file_id": fid})
            inbound = self._parse(entries[0][0], entries[0][1])
            if inbound:
                inbound.text = "\n".join(captions) if captions else "[media group]"
                inbound.media = media_items
                if not self.allowed_chats or inbound.peer_id in self.allowed_chats:
                    ready.append(inbound)

        # Flush text buffer after 300ms silence (allows time for complete pastes)
        expired_txt = [k for k, b in self._text_buf.items() if (now - b["ts"]) >= 0.3]
        for key in expired_txt:
            buf = self._text_buf.pop(key)
            buf["msg"].text = buf["text"]
            ready.append(buf["msg"])

        return ready

    # -- Message parsing --

    def _parse(self, msg: dict, raw_update: dict) -> InboundMessage | None:
        """
        Convert Telegram message format to InboundMessage.
        
        Handles:
        - Private chats (peer = user ID)
        - Group/supergroup chats (peer = chat ID)
        - Forum topics (peer = chat_id:topic:thread_id)
        """
        chat = msg.get("chat", {})
        chat_type = chat.get("type", "")
        chat_id = str(chat.get("id", ""))
        user_id = str(msg.get("from", {}).get("id", ""))
        text = msg.get("text", "") or msg.get("caption", "")
        if not text:
            return None

        thread_id = msg.get("message_thread_id")
        is_forum = chat.get("is_forum", False)
        is_group = chat_type in ("group", "supergroup")

        # Determine peer_id based on chat type
        if chat_type == "private":
            peer_id = user_id
        elif is_group and is_forum and thread_id is not None:
            # Forum topics get special peer_id format for targeting
            peer_id = f"{chat_id}:topic:{thread_id}"
        else:
            peer_id = chat_id

        return InboundMessage(
            text=text, sender_id=user_id, channel="telegram",
            account_id=self.account_id, peer_id=peer_id,
            is_group=is_group, raw=raw_update,
        )

    def receive(self) -> InboundMessage | None:
        """Poll for messages and return the first available."""
        msgs = self.poll()
        return msgs[0] if msgs else None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        """
        Send a message to a Telegram chat.
        
        Args:
            to: Chat ID, or "chat_id:topic:thread_id" for forum topics
            text: Message text (will be split if > 4096 chars)
        """
        # Parse forum topic if present
        chat_id, thread_id = to, None
        if ":topic:" in to:
            parts = to.split(":topic:")
            chat_id, thread_id = parts[0], int(parts[1]) if len(parts) > 1 else None
        
        ok = True
        # Split long messages to respect Telegram's limit
        for chunk in self._chunk(text):
            if not self._api("sendMessage", chat_id=chat_id, text=chunk,
                             message_thread_id=thread_id):
                ok = False
        return ok

    def _chunk(self, text: str) -> list[str]:
        """
        Split text into chunks respecting Telegram's 4096 char limit.
        Prefers splitting at newlines to keep messages readable.
        """
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text); break
            # Find last newline before limit for cleaner splits
            cut = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if cut <= 0:
                cut = self.MAX_MSG_LEN
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def close(self) -> None:
        """Close HTTP client on shutdown."""
        self._http.close()

# ---------------------------------------------------------------------------
# FeishuChannel -- webhook-based (also works with Lark/ByteDance)
# ---------------------------------------------------------------------------

class FeishuChannel(Channel):
    """
    Feishu/Lark channel using webhooks for receiving messages.
    
    Unlike Telegram's long-polling, Feishu uses webhooks (HTTP callbacks).
    Messages arrive via HTTP POST to a configured callback URL.
    
    Features:
    - Tenant access token management with auto-refresh
    - Support for text, post (rich text), and image messages
    - Group chat mention detection (only respond when bot is mentioned)
    - Encryption verification for secure webhooks
    
    Environment variables:
    - FEISHU_APP_ID, FEISHU_APP_SECRET (required)
    - FEISHU_ENCRYPT_KEY (optional, for encrypted webhooks)
    - FEISHU_BOT_OPEN_ID (optional, for mention detection)
    - FEISHU_IS_LARK (set to "true" for Lark/ByteDance variant)
    """
    name = "feishu"

    def __init__(self, account: ChannelAccount) -> None:
        """Initialize Feishu channel with app credentials."""
        if not HAS_HTTPX:
            raise RuntimeError("FeishuChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.app_id = account.config.get("app_id", "")
        self.app_secret = account.config.get("app_secret", "")
        self._encrypt_key = account.config.get("encrypt_key", "")
        self._bot_open_id = account.config.get("bot_open_id", "")
        is_lark = account.config.get("is_lark", False)
        
        # Choose API endpoint based on Feishu vs Lark variant
        self.api_base = ("https://open.larksuite.com/open-apis" if is_lark
                         else "https://open.feishu.cn/open-apis")
        
        # Token management for API calls
        self._tenant_token: str = ""
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _refresh_token(self) -> str:
        """
        Get a valid tenant access token, refreshing if necessary.
        
        Feishu tokens expire after ~2 hours, so we refresh proactively
        (300 seconds before expiry) to avoid mid-request failures.
        """
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        try:
            resp = self._http.post(
                f"{self.api_base}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Token error: {data.get('msg', '?')}{RESET}")
                return ""
            self._tenant_token = data.get("tenant_access_token", "")
            # Refresh 5 minutes before expiry
            self._token_expires_at = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        except Exception as exc:
            print(f"  {RED}[feishu] Token error: {exc}{RESET}")
            return ""

    def _bot_mentioned(self, event: dict) -> bool:
        """
        Check if the bot was mentioned in a group message.
        
        In group chats, we should only respond when explicitly mentioned
        to avoid noise. This checks various mention formats.
        """
        for m in event.get("message", {}).get("mentions", []):
            mid = m.get("id", {})
            if isinstance(mid, dict) and mid.get("open_id") == self._bot_open_id:
                return True
            if isinstance(mid, str) and mid == self._bot_open_id:
                return True
            if m.get("key") == self._bot_open_id:
                return True
        return False

    def _parse_content(self, message: dict) -> tuple[str, list]:
        """
        Parse Feishu message content into text and media.
        
        Supports:
        - text: Plain text messages
        - post: Rich text posts with title and content
        - image: Image attachments
        
        Returns:
            Tuple of (text, media_items)
        """
        msg_type = message.get("msg_type", "text")
        raw = message.get("content", "{}")
        try:
            content = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return "", []

        media: list[dict] = []
        if msg_type == "text":
            return content.get("text", ""), media
        if msg_type == "post":
            # Parse rich text post format
            texts: list[str] = []
            for lc in content.values():
                if not isinstance(lc, dict):
                    continue
                title = lc.get("title", "")
                if title:
                    texts.append(title)
                for para in lc.get("content", []):
                    for node in para:
                        tag = node.get("tag")
                        if tag == "text":
                            texts.append(node.get("text", ""))
                        elif tag == "a":
                            texts.append(node.get("text", "") + " " + node.get("href", ""))
            return "\n".join(texts), media
        if msg_type == "image":
            key = content.get("image_key", "")
            if key:
                media.append({"type": "image", "key": key})
            return "[image]", media
        return "", media

    def parse_event(self, payload: dict, token: str = "") -> InboundMessage | None:
        """
        Parse a Feishu event callback/webhook payload.
        
        This is called by the webhook handler (not by receive()).
        Handles:
        - URL verification challenge (for initial webhook setup)
        - Token verification (for encrypted webhooks)
        - Message events (text, post, image)
        
        Args:
            payload: The JSON payload from Feishu webhook
            token: The verification token from webhook URL
            
        Returns:
            InboundMessage if valid message, None otherwise
        """
        # Verify encryption token if configured
        if self._encrypt_key and token and token != self._encrypt_key:
            print(f"  {RED}[feishu] Token verification failed{RESET}")
            return None
        # Handle URL verification challenge (one-time on webhook setup)
        if "challenge" in payload:
            print_info(f"[feishu] Challenge: {payload['challenge']}")
            return None

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("open_id", sender.get("user_id", ""))
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"

        # In groups, only respond when mentioned
        if is_group and self._bot_open_id and not self._bot_mentioned(event):
            return None

        text, media = self._parse_content(message)
        if not text:
            return None

        return InboundMessage(
            text=text, sender_id=user_id, channel="feishu",
            account_id=self.account_id,
            peer_id=user_id if chat_type == "p2p" else chat_id,
            media=media, is_group=is_group, raw=payload,
        )

    def receive(self) -> InboundMessage | None:
        """
        Feishu uses webhooks, not polling.
        
        This returns None because messages come via parse_event()
        which should be called by the HTTP webhook handler.
        """
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        """Send a text message to a Feishu chat."""
        token = self._refresh_token()
        if not token:
            return False
        try:
            resp = self._http.post(
                f"{self.api_base}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": to, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Send: {data.get('msg', '?')}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"  {RED}[feishu] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        """Close HTTP client on shutdown."""
        self._http.close()

# ---------------------------------------------------------------------------
# Tools - Functions the AI can call during conversation
# ---------------------------------------------------------------------------

# Memory file location in workspace
MEMORY_FILE = WORKSPACE_DIR / "MEMORY.md"

def tool_memory_write(content: str) -> str:
    """
    Tool: Save a note to long-term memory (append to file).
    
    The AI uses this to remember important information across conversations.
    Stored as simple Markdown list items for easy searching.
    """
    print_tool("memory_write", f"{len(content)} chars")
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n- {content}\n")
        return f"Written to memory: {content[:80]}..."
    except Exception as exc:
        return f"Error: {exc}"

def tool_memory_search(query: str) -> str:
    """
    Tool: Search through saved memory notes.
    
    Case-insensitive substring search through the memory file.
    Returns up to 20 matching lines.
    """
    print_tool("memory_search", query)
    if not MEMORY_FILE.exists():
        return "Memory file is empty."
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        matches = [l for l in lines if query.lower() in l.lower()]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."
    except Exception as exc:
        return f"Error: {exc}"

# Tool definitions for Claude API (describe what tools exist and their schemas)
TOOLS = [
    {"name": "memory_write", "description": "Save a note to long-term memory.",
     "input_schema": {"type": "object", "required": ["content"],
                      "properties": {"content": {"type": "string",
                                                  "description": "The text to remember."}}}},
    {"name": "memory_search", "description": "Search through saved memory notes.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string",
                                               "description": "Search keyword."}}}},
]

# Map tool names to their Python handler functions
TOOL_HANDLERS: dict[str, Any] = {
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Route a tool call to its handler function.
    
    Called when Claude API returns a tool_use response.
    Returns the tool's output string to send back to the model.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"

# ---------------------------------------------------------------------------
# ChannelManager - Registry for all active channels
# ---------------------------------------------------------------------------

class ChannelManager:
    """
    Manages registered channels and their configurations.
    
    Provides a central registry for all active channel implementations.
    The agent loop queries this to find channels for sending responses.
    """
    def __init__(self) -> None:
        self.channels: dict[str, Channel] = {}  # name -> Channel instance
        self.accounts: list[ChannelAccount] = []  # All configured accounts

    def register(self, channel: Channel) -> None:
        """Register a channel instance."""
        self.channels[channel.name] = channel
        print_channel(f"  [+] Channel registered: {channel.name}")

    def list_channels(self) -> list[str]:
        """Get list of registered channel names."""
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def close_all(self) -> None:
        """Clean up all channels (close connections, etc.)."""
        for ch in self.channels.values():
            ch.close()

# ---------------------------------------------------------------------------
# Telegram background polling thread
# ---------------------------------------------------------------------------

def telegram_poll_loop(
    tg: TelegramChannel, queue: list, lock: threading.Lock, stop: threading.Event,
) -> None:
    """
    Background thread for polling Telegram messages.
    
    Runs in a separate thread to avoid blocking the main loop.
    Poll results are added to the queue for the main loop to process.
    
    Args:
        tg: TelegramChannel instance to poll
        queue: List to append incoming messages to
        lock: Threading lock for safe queue access
        stop: Event to signal thread shutdown
    """
    print_channel(f"  [telegram] Polling started for {tg.account_id}")
    while not stop.is_set():
        try:
            msgs = tg.poll()
            if msgs:
                with lock:
                    queue.extend(msgs)
        except Exception as exc:
            print(f"  {RED}[telegram] Poll error: {exc}{RESET}")
            stop.wait(5.0)  # Wait before retrying after error

# ---------------------------------------------------------------------------
# REPL commands - CLI commands for debugging/management
# ---------------------------------------------------------------------------

def handle_repl_command(cmd: str, mgr: ChannelManager) -> bool:
    """
    Handle slash commands from the CLI.
    
    These provide introspection and control without using the AI.
    
    Commands:
    - /channels: List registered channel types
    - /accounts: List configured bot accounts (tokens masked)
    - /help: Show available commands
    """
    cmd = cmd.strip().lower()
    if cmd == "/channels":
        for name in mgr.list_channels():
            print_channel(f"  - {name}")
        return True
    if cmd == "/accounts":
        for acc in mgr.accounts:
            masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
            print_channel(f"  - {acc.channel}/{acc.account_id}  token={masked}")
        return True
    if cmd in ("/help", "/h"):
        print_info("  /channels  /accounts  /help  quit/exit")
        return True
    return False

# ---------------------------------------------------------------------------
# Agent turn - Process one message through Claude and return response
# ---------------------------------------------------------------------------

def run_agent_turn(
    inbound: InboundMessage,
    conversations: dict[str, list[dict]],
    mgr: ChannelManager,
) -> None:
    """
    Process a single inbound message through the Claude API.
    
    Maintains conversation history per session (channel + account + peer).
    Handles multi-turn conversations with tool calls.
    
    Flow:
    1. Get or create conversation history for this session
    2. Add user message to history
    3. Send to Claude API with tools
    4. If tool call: execute tool, add result, continue (loop)
    5. If response: send back via channel, add to history
    
    Args:
        inbound: The incoming message to process
        conversations: Dict mapping session keys to message history
        mgr: ChannelManager for sending responses
    """
    # Get or create conversation history for this session
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    # Show typing indicator for Telegram (feedback that bot is processing)
    if inbound.channel == "telegram":
        tg = mgr.get("telegram")
        if isinstance(tg, TelegramChannel):
            tg.send_typing(inbound.peer_id.split(":topic:")[0])

    # Main API loop - handles multi-turn with tool calls
    while True:
        try:
            response = client.messages.create(
                model=MODEL_ID, max_tokens=8096,
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
        except Exception as exc:
            # On API error, try to recover by removing last assistant message
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return

        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason to determine next action
        if response.stop_reason == "end_turn":
            # Final response - extract text and send to user
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
                else:
                    print_assistant(text, inbound.channel)
            break
        elif response.stop_reason == "tool_use":
            # Claude wants to use a tool - execute and continue
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": process_tool_call(block.name, block.input),
                    })
            messages.append({"role": "user", "content": results})
        else:
            # Other stop reasons (e.g., max tokens) - send any text and exit
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
            break

# ---------------------------------------------------------------------------
# Main loop - Event-driven agent that processes messages from all channels
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    """
    Main event loop that processes messages from all channels.
    
    Architecture:
    - CLI channel is always registered for local testing
    - Telegram and Feishu are optional (require env vars)
    - Telegram runs in a background thread (polling)
    - Feishu uses webhooks (handled externally)
    
    The loop:
    1. Checks Telegram message queue (from background thread)
    2. Checks for CLI input (non-blocking when Telegram active)
    3. Processes each message through run_agent_turn()
    4. Handles REPL commands (/channels, /accounts, /help)
    5. Exits on "quit" or "exit"
    """
    # Initialize channel manager and register CLI (always available)
    mgr = ChannelManager()
    cli = CLIChannel()
    mgr.register(cli)

    # Telegram setup (optional - requires TELEGRAM_BOT_TOKEN)
    tg_channel: TelegramChannel | None = None
    stop_event = threading.Event()  # Signal for background thread to stop
    msg_queue: list[InboundMessage] = []  # Queue for Telegram messages
    q_lock = threading.Lock()  # Protect queue access
    tg_thread: threading.Thread | None = None

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if tg_token and HAS_HTTPX:
        tg_acc = ChannelAccount(
            channel="telegram", account_id="tg-primary", token=tg_token,
            config={"allowed_chats": os.getenv("TELEGRAM_ALLOWED_CHATS", "")},
        )
        mgr.accounts.append(tg_acc)
        tg_channel = TelegramChannel(tg_acc)
        mgr.register(tg_channel)
        # Start background polling thread
        tg_thread = threading.Thread(
            target=telegram_poll_loop, daemon=True,
            args=(tg_channel, msg_queue, q_lock, stop_event),
        )
        tg_thread.start()

    # Feishu setup (optional - requires FEISHU_APP_ID and FEISHU_APP_SECRET)
    fs_id = os.getenv("FEISHU_APP_ID", "").strip()
    fs_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if fs_id and fs_secret and HAS_HTTPX:
        fs_acc = ChannelAccount(
            channel="feishu", account_id="feishu-primary",
            config={
                "app_id": fs_id, "app_secret": fs_secret,
                "encrypt_key": os.getenv("FEISHU_ENCRYPT_KEY", ""),
                "bot_open_id": os.getenv("FEISHU_BOT_OPEN_ID", ""),
                "is_lark": os.getenv("FEISHU_IS_LARK", "").lower() in ("1", "true"),
            },
        )
        mgr.accounts.append(fs_acc)
        mgr.register(FeishuChannel(fs_acc))

    # Print startup banner
    print_info("=" * 60)
    print_info("  claw0  |  Section 04: Channels")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Channels: {', '.join(mgr.list_channels())}")
    print_info("  Commands: /channels /accounts /help  |  quit/exit")
    print_info("=" * 60)
    print()

    # Conversation history per session (keyed by channel+account+peer)
    conversations: dict[str, list[dict]] = {}

    # Main event loop
    while True:
        # Drain Telegram queue (messages from background thread)
        with q_lock:
            tg_msgs = msg_queue[:]
            msg_queue.clear()
        for m in tg_msgs:
            print_channel(f"\n  [telegram] {m.sender_id}: {m.text[:80]}")
            run_agent_turn(m, conversations, mgr)

        # CLI input handling
        # When Telegram is active, use non-blocking stdin with select
        if tg_channel:
            import select
            # Check if stdin has data (0.5s timeout)
            if not select.select([sys.stdin], [], [], 0.5)[0]:
                continue  # No input, loop back to check Telegram queue
            try:
                user_input = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
        else:
            # Blocking input when only CLI is active
            msg = cli.receive()
            if msg is None:
                break
            user_input = msg.text

        # Handle quit/exit commands
        if user_input.lower() in ("quit", "exit"):
            break
        # Handle REPL commands (slash commands)
        if user_input.startswith("/") and handle_repl_command(user_input, mgr):
            continue

        # Process regular message through agent
        run_agent_turn(
            InboundMessage(text=user_input, sender_id="cli-user",
                           channel="cli", account_id="cli-local", peer_id="cli-user"),
            conversations, mgr,
        )

    # Clean shutdown
    print(f"{DIM}Goodbye.{RESET}")
    stop_event.set()
    if tg_thread and tg_thread.is_alive():
        tg_thread.join(timeout=3.0)
    mgr.close_all()
    mgr.close_all()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Application entry point.
    
    Checks for required environment variables before starting.
    The only required variable is ANTHROPIC_API_KEY for Claude access.
    
    Optional variables:
    - MODEL_ID: Which Claude model to use (default: claude-sonnet-4-20250514)
    - TELEGRAM_BOT_TOKEN: Enable Telegram channel
    - FEISHU_APP_ID, FEISHU_APP_SECRET: Enable Feishu channel
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    agent_loop()

# Standard Python entry point pattern
if __name__ == "__main__":
    main()
