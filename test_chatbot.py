import os
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from streamlit.testing.v1 import AppTest

from chatbot_cloud import HISTORY_LIMIT, build_messages, get_model

APP_PATH = Path(__file__).with_name("chatbot_cloud.py")


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    @patch("chatbot_cloud.init_chat_model")
    def test_secrets_key_takes_precedence(self, init_model):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-test-key"}):
            with patch("chatbot_cloud.st.secrets", {"OPENAI_API_KEY": "secrets-test-key"}):
                get_model()
        self.assertEqual(init_model.call_args.kwargs["api_key"], "secrets-test-key")

    @patch("chatbot_cloud.init_chat_model")
    def test_environment_key_is_used_without_key_in_secrets(self, init_model):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-test-key"}):
            with patch("chatbot_cloud.st.secrets", {}):
                get_model()
        self.assertEqual(init_model.call_args.kwargs["api_key"], "environment-test-key")

    @patch("chatbot_cloud.init_chat_model")
    def test_missing_key_does_not_create_a_client(self, init_model):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch("chatbot_cloud.st.secrets", {}):
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    get_model()
        init_model.assert_not_called()

    def test_history_is_bounded_and_roles_are_preserved(self):
        history = [
            {"role": role, "content": f"Message {index}"}
            for index in range(12)
            for role in ("user", "assistant")
        ]
        messages = build_messages(history, "Next question")
        self.assertEqual(len(messages), HISTORY_LIMIT + 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], AIMessage)
        self.assertEqual(messages[1].content, history[-HISTORY_LIMIT]["content"])
        self.assertEqual(messages[-1].content, "Next question")

    @patch("langchain.chat_models.init_chat_model")
    def test_chat_remembers_turns_and_can_be_cleared(self, init_model):
        init_model.return_value.invoke.return_value = AIMessage(content="Hello!")
        app = AppTest.from_file(str(APP_PATH))
        app.secrets["OPENAI_API_KEY"] = "test-key-not-real"
        app.run()
        app.chat_input[0].set_value("Hello").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["messages"]), 2)
        init_model.assert_called_once()
        self.assertEqual(init_model.call_args.kwargs["model_provider"], "openai")
        self.assertEqual(init_model.call_args.args[0], "gpt-6-luna")
        self.assertEqual(init_model.call_args.kwargs["temperature"], 0)
        self.assertEqual(init_model.call_args.kwargs["top_p"], 1.0)
        self.assertEqual(init_model.call_args.kwargs["seed"], 1234)
        self.assertEqual(init_model.call_args.kwargs["reasoning_effort"], "none")

        app.chat_input[0].set_value("What did I say?").run()
        sent = init_model.return_value.invoke.call_args.args[0]
        self.assertEqual([message.content for message in sent[1:]], [
            "Hello", "Hello!", "What did I say?"
        ])
        self.assertEqual(len(app.session_state["messages"]), 4)
        app.button[0].click().run()
        self.assertEqual(app.session_state["messages"], [])
        self.assertFalse(app.exception)

    @patch("langchain.chat_models.init_chat_model")
    def test_failed_request_does_not_save_a_partial_turn(self, init_model):
        init_model.return_value.invoke.side_effect = RuntimeError("private details")
        app = AppTest.from_file(str(APP_PATH))
        app.secrets["OPENAI_API_KEY"] = "test-key-not-real"
        app.run()
        app.chat_input[0].set_value("Hello").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(len(app.error), 1)
        self.assertNotIn("private details", app.error[0].value)

    def tearDown(self):
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()