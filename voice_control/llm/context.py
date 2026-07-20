from typing import TYPE_CHECKING

from ..common.utils import get_logger

if TYPE_CHECKING:
    from .conversation import Conversation
    from .model import LLM


class DropOldestStrategy:
    """Trim oldest conversation turns when they exceed max_turns.

    Delegates the actual message removal to Conversation.trim_oldest() so
    the strategy stays focused on policy (when to trim) rather than
    mechanics (how to trim).
    """

    def __init__(self, max_turns: int = 20):
        self.logger = get_logger(__name__)
        self.max_turns = max_turns

    def trim(self, conversation: "Conversation", llm: "LLM") -> None:
        """Trim oldest messages if visible count exceeds max_turns."""
        excess = conversation.visible_count() - self.max_turns
        if excess <= 0:
            return
        total_cut = conversation.trim_oldest(excess, llm)
        self.logger.info(
            "Trimmed %d messages (~%d tokens)", excess, total_cut
        )
