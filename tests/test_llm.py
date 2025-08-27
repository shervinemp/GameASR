import unittest
from unittest.mock import patch, MagicMock
from voice_control.llm.model import NemotronLLM, QwenLLM, OllamaLLM
from voice_control.llm.conversation import Conversation

class TestLLM(unittest.TestCase):
    @patch('voice_control.llm.model.os.path.exists', return_value=True)
    @patch('voice_control.llm.model.Llama')
    def test_nemotron_llm(self, mock_llama, mock_exists):
        """
        Test the NemotronLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            {'choices': [{'delta': {'content': 'This is a test.'}}]}
        ])
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = NemotronLLM()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")

    @patch('voice_control.llm.model.ollama.Client')
    def test_empty_conversation(self, mock_ollama_client):
        """
        Test that the LLM returns an empty string for an empty conversation.
        """
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        mock_ollama_client.return_value = mock_client

        # Initialize the LLM
        llm = OllamaLLM()

        # Create an empty conversation
        conversation = Conversation()

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "")

    @patch('voice_control.llm.model.os.path.exists', return_value=True)
    @patch('voice_control.llm.model.Llama')
    def test_qwen_llm(self, mock_llama, mock_exists):
        """
        Test the QwenLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            {'choices': [{'delta': {'content': 'This is a test.'}}]}
        ])
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = QwenLLM()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")

    @patch('voice_control.llm.model.ollama.Client')
    def test_ollama_llm(self, mock_ollama_client):
        """
        Test the OllamaLLM.
        """
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([
            {'message': {'content': 'This is a test.'}}
        ])
        mock_ollama_client.return_value = mock_client

        # Initialize the LLM
        llm = OllamaLLM()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")
