# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "streamlit>=1.45,<2",
#     "langchain>=1.0,<2",
#     "langchain-openai>=1.0,<2",
#     "openai>=1.109,<3",
# ]
# ///

import os

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from streamlit.errors import StreamlitSecretNotFoundError

MODEL_NAME = "gpt-6-luna"
HISTORY_LIMIT = 20
SYSTEM_PROMPT = "You are a helpful assistant. Give clear, concise answers."


@st.cache_resource
def get_model():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", api_key)
    except StreamlitSecretNotFoundError:
        pass
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is missing or empty.")

    return init_chat_model(
        MODEL_NAME,
        model_provider="openai",
        temperature=0,
        top_p=1.0,
        seed=1234,
        reasoning_effort="none",
        api_key=api_key.strip(),
        max_retries=2,
        timeout=60,
    )


def build_messages(history, prompt):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for message in history[-HISTORY_LIMIT:]:
        message_type = HumanMessage if message["role"] == "user" else AIMessage
        messages.append(message_type(content=message["content"]))
    messages.append(HumanMessage(content=prompt))
    return messages

def main() -> None:
    st.set_page_config(page_title="OpenAI Chat", page_icon=":material/chat:")
    st.title("OpenAI Chat")
    st.caption(MODEL_NAME)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Conversation")
        if st.button("New chat", icon=":material/add:", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.download_button(
            "Download chat",
            data="\n\n".join(
                f"{message['role'].capitalize()}:\n{message['content']}"
                for message in st.session_state.messages
            ),
            file_name="conversation.txt",
            mime="text/plain",
            icon=":material/download:",
            disabled=not st.session_state.messages,
            use_container_width=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question", max_chars=4000)
    if not prompt or not prompt.strip():
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_model().invoke(
                    build_messages(st.session_state.messages, prompt)
                )
                answer = response.text
                if not answer.strip():
                    st.warning("No text response was returned. Try another question.")
                    return
                st.markdown(answer)
    except AuthenticationError:
        st.error("OpenAI authentication failed. Check your OPENAI_API_KEY.")
        return
    except PermissionDeniedError:
        st.error("Access denied. Check your OpenAI project permissions and model access.")
        return
    except (NotFoundError, BadRequestError):
        st.error("The model or request settings are unavailable. Check the model ID and supported parameters.")
        return
    except RateLimitError:
        st.error("OpenAI rate limit or quota exceeded. Check API billing and usage limits.")
        return
    except ValueError:
        st.error("Check OPENAI_API_KEY in Streamlit secrets or your environment, and verify the model settings.")
        return
    except Exception:
        st.error("The request failed. Check your configuration and try again.")
        return

    st.session_state.messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    st.rerun()


if __name__ == "__main__":
    main()
