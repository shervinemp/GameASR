from typing import TYPE_CHECKING
from ..common.utils import get_logger
from .conversation import Message

if TYPE_CHECKING:
    from .conversation import Conversation
    from .model import LLM


class ContextManager:
    """Manages the conversation context to keep token count within limits."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def manage_context(self, conversation: "Conversation", llm: "LLM"):
        """
        Prunes old messages from the conversation if the total token count
        exceeds the LLM's context limit.

        Args:
            conversation: The conversation object to manage.
            llm: The LLM instance (to access context limit and tokenizer).
        """
        # Determine the context limit
        # Default to a safe fallback if n_ctx is not present (e.g. for API models without explicit limit)
        # 4096 is a common default for many models
        context_limit = getattr(llm, "n_ctx", 4096)

        # Reserve some tokens for the new generation (e.g. max_tokens)
        max_gen_tokens = getattr(llm, "max_tokens", 512)

        # Effective limit for history
        history_limit = context_limit - max_gen_tokens

        # Calculate current tokens
        # We need to construct the prompt as it would be sent to the model to get accurate count.
        # However, exact formatting depends on the model.
        # We'll sum up tokens of individual messages as a reasonable approximation.

        total_tokens = 0
        message_tokens = []

        # Access raw messages to use caching
        all_messages = conversation._messages.iter_raw()

        # Calculate tokens for all messages
        for msg in all_messages:
            if msg.token_count is None:
                # Add some overhead for role/formatting (heuristic)
                msg.token_count = llm.count_tokens(msg.content) + 4

            tokens = msg.token_count
            message_tokens.append(tokens)
            total_tokens += tokens

        if total_tokens <= history_limit:
            return

        self.logger.info(f"Context usage ({total_tokens}) exceeds limit ({history_limit}). Pruning...")

        # Prune until we are under the limit
        # We must keep the system message if it exists (usually index 0? No, system message is separate in Conversation)
        # Conversation has _system field.
        # _messages are user/assistant/tool messages.

        # Start pruning from the beginning of _messages
        # We update cutoff_idx

        current_tokens = total_tokens
        new_cutoff = 0

        for i, tokens in enumerate(message_tokens):
            if current_tokens <= history_limit:
                break

            # Remove this message from the count
            current_tokens -= tokens
            new_cutoff = i + 1

        # Update cutoff_idx
        # conversation._cutoff_idx is an integer
        if new_cutoff > conversation._cutoff_idx:
            conversation.cutoff_idx = new_cutoff
            self.logger.info(f"Pruned {new_cutoff} messages. New context usage: {current_tokens}")
