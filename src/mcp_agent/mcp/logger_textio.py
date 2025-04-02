"""
Utilities for MCP stdio client integration with our logging system.
"""

import io
import sys
from typing import TextIO

from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class LoggerTextIO(TextIO):
    """
    A TextIO implementation that logs to our application logger.
    This implements the full TextIO interface as specified by Python.

    Args:
        server_name: The name of the server to include in logs
    """

    def __init__(self, server_name: str) -> None:
        super().__init__()
        self.server_name = server_name
        # Use a StringIO for buffering
        self._buffer = io.StringIO()
        # Keep track of complete and partial lines
        self._line_buffer = ""

    def write(self, s: str) -> int:
        """
        Write data to our buffer and log any complete lines.
        """
        if not s:
            return 0

        # Handle line buffering for clean log output
        text = self._line_buffer + s
        lines = text.split("\n")

        # If the text ends with a newline, the last line is complete
        if text.endswith("\n"):
            complete_lines = lines
            self._line_buffer = ""
        else:
            # Otherwise, the last line is incomplete
            complete_lines = lines[:-1]
            self._line_buffer = lines[-1]

        # Log complete lines but at debug level instead of info to prevent console spam
        for line in complete_lines:
            if line.strip():  # Only log non-empty lines
                logger.debug(f"{self.server_name} (stderr): {line}")

        # Always write to the underlying buffer
        return self._buffer.write(s)

    def flush(self) -> None:
        """Flush the internal buffer."""
        self._buffer.flush()

    def close(self) -> None:
        """Close the stream."""
        # Log any remaining content in the line buffer
        if self._line_buffer and self._line_buffer.strip():
            logger.debug(f"{self.server_name} (stderr): {self._line_buffer}")
        self._buffer.close()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        """
        Return a file descriptor for this stream.
        We use sys.stderr's fileno since TextIO is expected to return a real file descriptor.
        """
        return sys.stderr.fileno()


def get_stderr_handler(server_name: str) -> TextIO:
    """
    Get a stderr handler that routes MCP server errors to our logger.

    Args:
        server_name: The name of the server to include in logs

    Returns:
        A TextIO object that can be used as stderr by MCP
    """
    return LoggerTextIO(server_name)
