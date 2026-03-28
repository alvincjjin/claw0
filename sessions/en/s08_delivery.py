"""
Section 08: Delivery
"Write to disk first, then try to send"

This module implements a RELIABLE DELIVERY QUEUE with the following guarantees:
- Write-ahead logging: messages written to disk before delivery attempt
- Crash recovery: queue survives process restarts
- Exponential backoff: failed deliveries retry with increasing delays
- At-least-once delivery: messages may be delivered multiple times on failure

MESSAGE FLOW:
    Agent Reply / Heartbeat / Cron
              |
        chunk_message()       -- split by platform limits (Telegram: 4096, Discord: 2000)
              |
        DeliveryQueue.enqueue()  -- write to disk atomically (write-ahead log)
              |
        DeliveryRunner (background thread) -- daemon, ticks every 1s
              |
         deliver_fn(channel, to, text)
            /     \
         success    failure
           |           |
         ack()      fail() + backoff
           |           |
         delete      retry or move_to_failed/

CRASH RECOVERY:
- On startup, DeliveryRunner scans queue_dir/ and failed/ directories
- All pending entries are re-queued for delivery
- This ensures no messages are lost if the process crashes

EXPONENTIAL BACKOFF:
    Retry 1: wait 5s    (±20% jitter)
    Retry 2: wait 25s   (±20% jitter)
    Retry 3: wait 2min  (±20% jitter)
    Retry 4: wait 10min (±20% jitter)
    Retry 5: move to failed/

After 5 failed attempts, message is moved to failed/ directory for manual inspection.

QUEUE FILE FORMAT:
    {
      "id": "abc123def456",
      "channel": "telegram",
      "to": "chat123",
      "text": "Hello world",
      "retry_count": 2,
      "last_error": "Connection timeout",
      "enqueued_at": 1699999999.123,
      "next_retry_at": 1700000039.456
    }

Usage:
    cd claw0
    python sessions/en/s08_delivery.py

Required in .env:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Standard library for JSON, OS, randomness, threading, time, UUIDs
import json
import os
import random
import sys
import threading
import time
import uuid

# Data classes for structured data, Path for file paths
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Third-party: load environment variables, Anthropic SDK
from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load environment variables from .env file in project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# Model configuration (default: Claude Sonnet 4)
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

# Anthropic client for LLM calls
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# Directory structure for delivery queue persistence
# workspace/delivery-queue/    <- pending messages (JSON files)
# workspace/delivery-queue/failed/  <- permanently failed messages
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"
QUEUE_DIR = WORKSPACE_DIR / "delivery-queue"
FAILED_DIR = QUEUE_DIR / "failed"

# Exponential backoff schedule (milliseconds)
# Retry 1->2 uses index 0 (5s), retry 2->3 uses index 1 (25s), etc.
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]  # [5s, 25s, 2min, 10min]
MAX_RETRIES = 5  # After this many failures, move to failed/

# Base system prompt for the AI assistant
SYSTEM_PROMPT = (
    "You are Luna, a warm and curious AI companion. "
    "Keep replies concise and helpful. "
    "Use memory_write to save important facts. "
    "Use memory_search to recall past context."
)

# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------
# Color codes: ANSI escape sequences for colored terminal output
# Usage: print(f"{CYAN}text{RESET}")
CYAN = "\033[36m"     # User input prompt
GREEN = "\033[32m"    # Assistant messages
YELLOW = "\033[33m"  # Warnings
DIM = "\033[2m"       # Dimmed/info text
RESET = "\033[0m"     # Reset to default color
BOLD = "\033[1m"      # Bold text
MAGENTA = "\033[35m"  # Heartbeat messages
RED = "\033[31m"      # Errors
BLUE = "\033[34m"     # Delivery messages
ORANGE = "\033[38;5;208m"  # Extra highlight


def colored_prompt() -> str:
    """Return the styled prompt shown before user input."""
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    """Print LLM response in green."""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    """Print informational message in dim gray."""
    print(f"{DIM}{text}{RESET}")


def print_delivery(text: str) -> None:
    """Print delivery service message in blue."""
    print(f"  {BLUE}[delivery]{RESET} {text}")


def print_warn(text: str) -> None:
    """Print warning message in yellow."""
    print(f"  {YELLOW}[warn]{RESET} {text}")


def print_error(text: str) -> None:
    """Print error message in red."""
    print(f"  {RED}[error]{RESET} {text}")


# ---------------------------------------------------------------------------
# 1. QueuedDelivery -- queue entry data structure
# ---------------------------------------------------------------------------
# Represents a single message waiting to be delivered.
# Stored as JSON files in the queue directory for crash recovery.

@dataclass
class QueuedDelivery:
    """A single delivery entry in the queue."""
    id: str           # Unique identifier (12 hex chars)
    channel: str      # Target channel (telegram, discord, etc.)
    to: str           # Recipient identifier (chat_id, user_id, etc.)
    text: str         # Message content
    retry_count: int = 0          # Number of failed delivery attempts
    last_error: str | None = None  # Last error message if failed
    enqueued_at: float = field(default_factory=time.time)  # Unix timestamp
    next_retry_at: float = 0.0    # Next retry time (0 = deliver immediately)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "channel": self.channel,
            "to": self.to,
            "text": self.text,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "enqueued_at": self.enqueued_at,
            "next_retry_at": self.next_retry_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueuedDelivery":
        """Deserialize from dictionary loaded from JSON."""
        return QueuedDelivery(
            id=data["id"],
            channel=data["channel"],
            to=data["to"],
            text=data["text"],
            retry_count=data.get("retry_count", 0),
            last_error=data.get("last_error"),
            enqueued_at=data.get("enqueued_at", 0.0),
            next_retry_at=data.get("next_retry_at", 0.0),
        )


def compute_backoff_ms(retry_count: int) -> int:
    """Calculate backoff delay for given retry count.

    Uses exponential backoff with +/- 20% random jitter to prevent
    "thundering herd" where many clients retry at exactly the same time.

    Args:
        retry_count: Number of previous failed attempts (1-indexed)

    Returns:
        Delay in milliseconds for next retry
    """
    if retry_count <= 0:
        return 0
    # Map retry_count to backoff index (retry 1 -> index 0, retry 2 -> index 1, etc.)
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    base = BACKOFF_MS[idx]
    # Jitter: +/- 20% of base delay
    jitter = random.randint(-base // 5, base // 5)
    return max(0, base + jitter)


# ---------------------------------------------------------------------------
# 2. DeliveryQueue -- disk-persisted reliable delivery queue
# ---------------------------------------------------------------------------
# Implements write-ahead logging: messages are written to disk BEFORE
# attempting delivery. This ensures no message loss on crash.
#
# CRASH SAFETY:
# - Atomic file writes: write to temp file, then os.replace() to final name
# - fsync: flush OS buffer to disk for durability
# - Recovery: load_pending() scans disk on startup to recover undelivered messages
#
# FILE STRUCTURE:
#   delivery-queue/
#     abc123.json      <- pending delivery
#     def456.json      <- pending delivery
#   delivery-queue/failed/
#     ghi789.json      <- permanently failed (after MAX_RETRIES)


class DeliveryQueue:
    """Disk-persisted queue for reliable message delivery."""

    def __init__(self, queue_dir: Path | None = None):
        self.queue_dir = queue_dir or QUEUE_DIR
        self.failed_dir = self.queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # Protects concurrent access to queue

    def enqueue(self, channel: str, to: str, text: str) -> str:
        """Create a queue entry and atomically write it to disk.

        This is the main entry point for adding messages to the queue.
        Returns a unique delivery_id that can be used to track the message.

        Args:
            channel: Target platform (telegram, discord, etc.)
            to: Recipient identifier
            text: Message content

        Returns:
            12-character hex delivery_id
        """
        delivery_id = uuid.uuid4().hex[:12]
        entry = QueuedDelivery(
            id=delivery_id,
            channel=channel,
            to=to,
            text=text,
            enqueued_at=time.time(),
            next_retry_at=0.0,  # 0 = deliver immediately
        )
        self._write_entry(entry)
        return delivery_id

    def _write_entry(self, entry: QueuedDelivery) -> None:
        """Atomic write via tmp file + os.replace().

        This ensures crash safety: either the old file exists or the new one,
        never a partial/corrupt file.
        """
        final_path = self.queue_dir / f"{entry.id}.json"
        tmp_path = self.queue_dir / f".tmp.{os.getpid()}.{entry.id}.json"
        data = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        os.replace(str(tmp_path), str(final_path))  # Atomic on POSIX

    def _read_entry(self, delivery_id: str) -> QueuedDelivery | None:
        """Load a single entry from the queue by ID."""
        file_path = self.queue_dir / f"{delivery_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QueuedDelivery.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def ack(self, delivery_id: str) -> None:
        """Mark delivery as successful - remove from queue."""
        file_path = self.queue_dir / f"{delivery_id}.json"
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass  # Already deleted or never existed

    def fail(self, delivery_id: str, error: str) -> None:
        """Mark delivery as failed - increment retry count and schedule next attempt.

        If MAX_RETRIES exceeded, moves entry to failed/ directory.
        """
        entry = self._read_entry(delivery_id)
        if entry is None:
            return
        entry.retry_count += 1
        entry.last_error = error
        if entry.retry_count >= MAX_RETRIES:
            self.move_to_failed(delivery_id)
            return
        # Schedule next retry with exponential backoff
        backoff_ms = compute_backoff_ms(entry.retry_count)
        entry.next_retry_at = time.time() + backoff_ms / 1000.0
        self._write_entry(entry)

    def move_to_failed(self, delivery_id: str) -> None:
        """Move a permanently failed message to the failed/ directory."""
        src = self.queue_dir / f"{delivery_id}.json"
        dst = self.failed_dir / f"{delivery_id}.json"
        try:
            os.replace(str(src), str(dst))
        except FileNotFoundError:
            pass  # Already moved or never existed

    def load_pending(self) -> list[QueuedDelivery]:
        """Scan queue directory and load all pending entries.

        Returns:
            List sorted by enqueued_at (oldest first)
        """
        entries: list[QueuedDelivery] = []
        if not self.queue_dir.exists():
            return entries
        for file_path in self.queue_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # Skip corrupt files
        entries.sort(key=lambda e: e.enqueued_at)
        return entries

    def load_failed(self) -> list[QueuedDelivery]:
        """Load all permanently failed entries from failed/ directory."""
        entries: list[QueuedDelivery] = []
        if not self.failed_dir.exists():
            return entries
        for file_path in self.failed_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        entries.sort(key=lambda e: e.enqueued_at)
        return entries

    def retry_failed(self) -> int:
        """Move all failed entries back to the queue with reset retry_count.

        Used for manual recovery after fixing the underlying issue.

        Returns:
            Number of entries retried
        """
        count = 0
        if not self.failed_dir.exists():
            return count
        for file_path in self.failed_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = QueuedDelivery.from_dict(data)
                entry.retry_count = 0
                entry.last_error = None
                entry.next_retry_at = 0.0
                self._write_entry(entry)
                file_path.unlink()
                count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return count


# ---------------------------------------------------------------------------
# 3. Channel-aware message chunking
# ---------------------------------------------------------------------------
# Different platforms have different message size limits.
# This module ensures messages are split appropriately.

# Platform character limits
# Telegram: 4096 chars (or 1024 for captions with media)
# Discord: 2000 chars per message
# WhatsApp: 4096 chars
CHANNEL_LIMITS: dict[str, int] = {
    "telegram": 4096,
    "telegram_caption": 1024,
    "discord": 2000,
    "whatsapp": 4096,
    "default": 4096,
}


def chunk_message(text: str, channel: str = "default") -> list[str]:
    """Split message into platform-appropriate chunks.

    Algorithm:
    1. Try to group paragraphs together under the limit
    2. If a single paragraph exceeds the limit, hard-split it

    Args:
        text: Message text to split
        channel: Target platform (determines size limit)

    Returns:
        List of chunks, each within platform limits
    """
    if not text:
        return []
    limit = CHANNEL_LIMITS.get(channel, CHANNEL_LIMITS["default"])
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    # Split by paragraph (double newline), try to group under limit
    for para in text.split("\n\n"):
        if chunks and len(chunks[-1]) + len(para) + 2 <= limit:
            # Fit in current chunk
            chunks[-1] += "\n\n" + para
        else:
            # Start new chunk, hard-split if paragraph exceeds limit
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            if para:
                chunks.append(para)

    return chunks or [text[:limit]]


# ---------------------------------------------------------------------------
# 4. DeliveryRunner -- background delivery thread
# ---------------------------------------------------------------------------
# Runs as a daemon thread, continuously processing the delivery queue.
# Ticks every 1 second to check for entries ready for delivery.
#
# CRASH RECOVERY:
# On startup, scans the queue directory to recover any pending messages
# that were not delivered before a previous crash.

class DeliveryRunner:
    """Background thread that processes the delivery queue."""

    def __init__(
        self,
        queue: DeliveryQueue,
        deliver_fn: Callable[[str, str, str], None],
    ):
        self.queue = queue
        self.deliver_fn = deliver_fn  # Function that actually sends the message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics for monitoring
        self.total_attempted = 0
        self.total_succeeded = 0
        self.total_failed = 0

    def start(self) -> None:
        """Run recovery scan, then start background delivery thread."""
        self._recovery_scan()  # Load any pending entries from disk
        self._thread = threading.Thread(
            target=self._background_loop,
            daemon=True,  # Won't block program exit
            name="delivery-runner",
        )
        self._thread.start()

    def _recovery_scan(self) -> None:
        """Scan disk for pending/failed entries on startup.

        This ensures no messages are lost if the process crashed
        with undelivered messages in the queue.
        """
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        parts = []
        if pending:
            parts.append(f"{len(pending)} pending")
        if failed:
            parts.append(f"{len(failed)} failed")
        if parts:
            print_delivery(f"Recovery: {', '.join(parts)}")
        else:
            print_delivery("Recovery: queue is clean")

    def _background_loop(self) -> None:
        """Main delivery loop - runs until stop() is called."""
        while not self._stop_event.is_set():
            try:
                self._process_pending()
            except Exception as exc:
                print_error(f"Delivery loop error: {exc}")
            self._stop_event.wait(timeout=1.0)  # Tick every 1 second

    def _process_pending(self) -> None:
        """Process all entries whose next_retry_at <= now.

        For each entry:
        - If delivery succeeds: ack() to remove from queue
        - If delivery fails: fail() to increment retry count and schedule next attempt
        """
        pending = self.queue.load_pending()
        now = time.time()

        for entry in pending:
            if self._stop_event.is_set():
                break
            # Skip entries not yet due for retry
            if entry.next_retry_at > now:
                continue

            self.total_attempted += 1
            try:
                # Attempt delivery via the provided function
                self.deliver_fn(entry.channel, entry.to, entry.text)
                self.queue.ack(entry.id)  # Success - remove from queue
                self.total_succeeded += 1
            except Exception as exc:
                # Failure - record error and schedule retry
                error_msg = str(exc)
                self.queue.fail(entry.id, error_msg)
                self.total_failed += 1
                retry_info = f"retry {entry.retry_count + 1}/{MAX_RETRIES}"
                if entry.retry_count + 1 >= MAX_RETRIES:
                    print_warn(
                        f"Delivery {entry.id[:8]}... -> failed/ ({retry_info}): {error_msg}"
                    )
                else:
                    backoff = compute_backoff_ms(entry.retry_count + 1)
                    print_warn(
                        f"Delivery {entry.id[:8]}... failed ({retry_info}), "
                        f"next retry in {backoff / 1000:.0f}s: {error_msg}"
                    )

    def stop(self) -> None:
        """Stop the delivery thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_stats(self) -> dict:
        """Return current delivery statistics."""
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        return {
            "pending": len(pending),
            "failed": len(failed),
            "total_attempted": self.total_attempted,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
        }


# ---------------------------------------------------------------------------
# 5. MockDeliveryChannel -- simulated delivery channel
# ---------------------------------------------------------------------------
# Used for testing the delivery system without actual network calls.
# Allows configurable failure rate to test retry logic.

class MockDeliveryChannel:
    """Simulated delivery channel for testing."""

    def __init__(self, name: str, fail_rate: float = 0.0):
        self.name = name
        self.fail_rate = fail_rate  # 0.0 = always succeed, 1.0 = always fail
        self.sent: list[dict] = []  # Track all sent messages

    def send(self, to: str, text: str) -> None:
        """Simulate sending a message.

        Raises ConnectionError at the configured fail_rate to simulate
        network failures.
        """
        if random.random() < self.fail_rate:
            raise ConnectionError(
                f"[{self.name}] Simulated delivery failure to {to}"
            )
        self.sent.append({"to": to, "text": text, "time": time.time()})
        preview = text[:60].replace("\n", " ")
        print_delivery(f"[{self.name}] -> {to}: {preview}...")

    def set_fail_rate(self, rate: float) -> None:
        """Update the failure rate (0.0 to 1.0)."""
        self.fail_rate = max(0.0, min(1.0, rate))


# ---------------------------------------------------------------------------
# 6. Soul + Memory (simplified, with tool integration)
# ---------------------------------------------------------------------------
# SoulSystem: Loads the agent's personality from SOUL.md
# MemoryStore: Persistent JSONL store for long-term memory
# Tools: memory_write and memory_search for the LLM to use

class SoulSystem:
    """Loads agent personality/prompt from SOUL.md."""

    def __init__(self):
        soul_path = WORKSPACE_DIR / "SOUL.md"
        if soul_path.exists():
            self.personality = soul_path.read_text(encoding="utf-8")
        else:
            self.personality = ""

    def get_system_prompt(self) -> str:
        """Build full system prompt: personality + base instructions."""
        base = SYSTEM_PROMPT
        if self.personality:
            base = f"{self.personality}\n\n{base}"
        return base


class MemoryStore:
    """Simple JSONL-based memory store."""

    def __init__(self):
        self.memory_file = WORKSPACE_DIR / "memory.jsonl"
        if not self.memory_file.exists():
            self.memory_file.touch()

    def write(self, content: str) -> str:
        """Append a new memory entry to the store."""
        entry = {"content": content, "time": time.time()}
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return f"Saved: {content[:50]}"

    def search(self, query: str) -> str:
        """Search memories for lines containing the query."""
        if not self.memory_file.exists():
            return "No memories found."
        query_lower = query.lower()
        results: list[str] = []
        with open(self.memory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if query_lower in entry.get("content", "").lower():
                        results.append(entry["content"])
                except json.JSONDecodeError:
                    continue
        if not results:
            return "No memories found."
        return "\n".join(f"- {r}" for r in results[-5:])


# Tool definitions for Claude API
TOOLS = [
    {
        "name": "memory_write",
        "description": "Save an important fact or preference to long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or preference to remember.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search long-term memory for relevant facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# 7. HeartbeatRunner -- timer that enqueues through DeliveryQueue
# ---------------------------------------------------------------------------
# Generates periodic heartbeat messages that go through the delivery queue.
# Demonstrates how the delivery system handles automated (non-user) messages.
#
# Unlike s07 which used lane_lock to yield to user input, this version
# simply enqueues to the delivery queue which handles reliability.

class HeartbeatRunner:
    """Periodically generates heartbeat messages via the delivery queue."""

    def __init__(
        self,
        queue: DeliveryQueue,
        channel: str,
        to: str,
        interval: float = 60.0,
    ):
        self.queue = queue
        self.channel = channel
        self.to = to
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lane_lock = threading.Lock()
        self.last_run: float = 0.0
        self.run_count: int = 0
        self.enabled: bool = False

    def start(self) -> None:
        """Start the heartbeat daemon thread."""
        self.enabled = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="heartbeat-runner",
        )
        self._thread.start()

    def _loop(self) -> None:
        """Main heartbeat loop - ticks every interval seconds."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.interval)
            if self._stop_event.is_set():
                break
            if not self.enabled:
                continue
            self.trigger()

    def trigger(self) -> None:
        """Generate heartbeat text and enqueue for delivery.

        The message goes through the reliable delivery queue,
        so it will be retried if delivery fails.
        """
        with self._lane_lock:
            self.last_run = time.time()
            self.run_count += 1
            heartbeat_text = (
                f"[Heartbeat #{self.run_count}] "
                f"System check at {time.strftime('%H:%M:%S')} -- all OK."
            )
            chunks = chunk_message(heartbeat_text, self.channel)
            for chunk in chunks:
                self.queue.enqueue(self.channel, self.to, chunk)
            print_info(f"  {MAGENTA}[heartbeat]{RESET} triggered #{self.run_count}")

    def stop(self) -> None:
        """Stop the heartbeat thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_status(self) -> dict:
        """Return current heartbeat status."""
        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "run_count": self.run_count,
            "last_run": time.strftime(
                "%H:%M:%S", time.localtime(self.last_run)
            ) if self.last_run else "never",
        }


# ---------------------------------------------------------------------------
# 8. Agent loop + REPL
# ---------------------------------------------------------------------------
# Main event loop that combines:
# - User interaction via REPL
# - LLM agent for responses
# - Delivery queue for reliable output
# - Background heartbeat and delivery runner

def process_tool_call(
    tool_name: str,
    tool_input: dict,
    memory: MemoryStore,
) -> str:
    """Handle tool calls from the LLM."""
    if tool_name == "memory_write":
        return memory.write(tool_input["content"])
    elif tool_name == "memory_search":
        return memory.search(tool_input["query"])
    return f"Error: Unknown tool '{tool_name}'"


def handle_repl_command(
    cmd: str,
    queue: DeliveryQueue,
    runner: DeliveryRunner,
    heartbeat: HeartbeatRunner,
    mock_channel: MockDeliveryChannel,
) -> bool:
    """Handle REPL commands. Returns True if command was handled."""
    # /queue - show pending deliveries
    if cmd == "/queue":
        pending = queue.load_pending()
        if not pending:
            print_info("  Queue is empty.")
            return True
        print_info(f"  Pending deliveries ({len(pending)}):")
        now = time.time()
        for entry in pending:
            wait = ""
            if entry.next_retry_at > now:
                remaining = entry.next_retry_at - now
                wait = f", wait {remaining:.0f}s"
            preview = entry.text[:40].replace("\n", " ")
            print_info(
                f"    {entry.id[:8]}... "
                f"retry={entry.retry_count}{wait} "
                f'"{preview}"'
            )
        return True

    # /failed - show permanently failed deliveries
    if cmd == "/failed":
        failed = queue.load_failed()
        if not failed:
            print_info("  No failed deliveries.")
            return True
        print_info(f"  Failed deliveries ({len(failed)}):")
        for entry in failed:
            preview = entry.text[:40].replace("\n", " ")
            err = entry.last_error or "unknown"
            print_info(
                f"    {entry.id[:8]}... "
                f"retries={entry.retry_count} "
                f'error="{err[:30]}" '
                f'"{preview}"'
            )
        return True

    # /retry - move failed entries back to queue
    if cmd == "/retry":
        count = queue.retry_failed()
        print_info(f"  Moved {count} entries back to queue.")
        return True

    # /simulate-failure - toggle 50% failure rate for testing
    if cmd == "/simulate-failure":
        if mock_channel.fail_rate > 0:
            mock_channel.set_fail_rate(0.0)
            print_info(f"  {mock_channel.name} fail rate -> 0% (reliable)")
        else:
            mock_channel.set_fail_rate(0.5)
            print_info(f"  {mock_channel.name} fail rate -> 50% (unreliable)")
        return True

    # /heartbeat - show heartbeat status
    if cmd == "/heartbeat":
        status = heartbeat.get_status()
        print_info(f"  Heartbeat: enabled={status['enabled']}, "
                   f"interval={status['interval']}s, "
                   f"runs={status['run_count']}, "
                   f"last={status['last_run']}")
        return True

    # /trigger - manually trigger heartbeat
    if cmd == "/trigger":
        heartbeat.trigger()
        return True

    # /stats - show delivery statistics
    if cmd == "/stats":
        stats = runner.get_stats()
        print_info(f"  Delivery stats: "
                   f"pending={stats['pending']}, "
                   f"failed={stats['failed']}, "
                   f"attempted={stats['total_attempted']}, "
                   f"succeeded={stats['total_succeeded']}, "
                   f"errors={stats['total_failed']}")
        return True

    return False


def agent_loop() -> None:
    """Main REPL loop with integrated delivery queue and background services.

    Flow:
    1. Initialize Soul, Memory, DeliveryQueue, DeliveryRunner, Heartbeat
    2. Start background threads (delivery runner, heartbeat)
    3. Loop: read user input -> call LLM -> enqueue response to delivery queue
    4. On exit: stop background threads (queue state preserved on disk)
    """
    # Initialize components
    soul = SoulSystem()
    memory = MemoryStore()
    system_prompt = soul.get_system_prompt()

    # Mock delivery channel (simulates actual sending)
    mock_channel = MockDeliveryChannel("console", fail_rate=0.0)
    default_channel = "console"
    default_to = "user"

    # Delivery queue (disk-persisted)
    queue = DeliveryQueue()

    # Delivery function that sends via mock channel
    def deliver_fn(channel: str, to: str, text: str) -> None:
        mock_channel.send(to, text)

    # Start background delivery runner
    runner = DeliveryRunner(queue, deliver_fn)
    runner.start()

    # Start background heartbeat (generates periodic messages)
    heartbeat = HeartbeatRunner(
        queue=queue,
        channel=default_channel,
        to=default_to,
        interval=120.0,  # Every 2 minutes
    )
    heartbeat.start()

    # Conversation history
    messages: list[dict] = []

    # Print startup banner
    print_info("=" * 60)
    print_info("  claw0  |  Section 08: Delivery")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Queue: {QUEUE_DIR}")
    print_info("  Commands:")
    print_info("    /queue             - show pending deliveries")
    print_info("    /failed            - show failed deliveries")
    print_info("    /retry             - retry all failed")
    print_info("    /simulate-failure  - toggle 50% failure rate")
    print_info("    /heartbeat         - heartbeat status")
    print_info("    /trigger           - manually trigger heartbeat")
    print_info("    /stats             - delivery statistics")
    print_info("  Type 'quit' or 'exit' to leave.")
    print_info("=" * 60)
    print()

    # Main REPL loop
    while True:
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

        # Handle REPL commands
        if user_input.startswith("/"):
            if handle_repl_command(
                user_input, queue, runner, heartbeat, mock_channel
            ):
                continue
            print_info(f"  Unknown command: {user_input}")
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Agent inner loop (handles tool calls)
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                # Rollback to last user message on error
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Handle different stop reasons
            if response.stop_reason == "end_turn":
                # Normal text response - print and enqueue for delivery
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                    # Enqueue response to delivery queue (reliable delivery)
                    chunks = chunk_message(assistant_text, default_channel)
                    for chunk in chunks:
                        queue.enqueue(default_channel, default_to, chunk)
                break

            elif response.stop_reason == "tool_use":
                # LLM wants to call a tool - execute and continue
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = process_tool_call(block.name, block.input, memory)
                    print_info(f"  {DIM}[tool: {block.name}]{RESET}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            else:
                # Other stop reasons - treat as text response
                print_info(f"[stop_reason={response.stop_reason}]")
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                    chunks = chunk_message(assistant_text, default_channel)
                    for chunk in chunks:
                        queue.enqueue(default_channel, default_to, chunk)
                break

    # Cleanup: stop background threads
    heartbeat.stop()
    runner.stop()
    print_info("Delivery runner stopped. Queue state preserved on disk.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Validate API key and start the agent loop."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
