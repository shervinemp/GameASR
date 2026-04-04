import unittest
from unittest.mock import patch, MagicMock
from voice_control.llm.model import NemotronMini, Qwen3, Ollama
from voice_control.llm.conversation import Conversation


class TestLLM(unittest.TestCase):
    @patch("voice_control.llm.model.os.path.exists", return_value=True)
    @patch("voice_control.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voice_control.llm.model.Llama", create=True)
    def test_nemotron_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test the NemotronLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = NemotronMini()
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    @patch("ollama.Client")
    def test_empty_conversation(self, mock_ollama_client):
        """
        Test that the LLM returns an empty string for an empty conversation.
        """
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        mock_ollama_client.return_value = mock_client

        # Initialize the LLM
        llm = Ollama(model="test")

        # Create an empty conversation
        conversation = Conversation()

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "")

    @patch("voice_control.llm.model.os.path.exists", return_value=True)
    @patch("voice_control.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voice_control.llm.model.Llama", create=True)
    def test_qwen_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test the QwenLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = Qwen3()
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    @patch("ollama.Client")
    def test_ollama_llm(self, mock_ollama_client):
        """
        Test the OllamaLLM.
        """
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(
            [{"message": {"content": "This is a test."}}]
        )
        mock_ollama_client.return_value = mock_client

        # Initialize the LLM
        llm = Ollama(model="test")

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")
