# Streamlit + LangChain OpenAI Chatbot

A small learning demo: type a question, receive an OpenAI reply, and ask follow-up
questions. Includes per-session history, a new-chat button, and a text download.
There is no database, agent, retrieval pipeline, or LangChain memory framework.

## Files

| File | Purpose |
| --- | --- |
| `chatbot_cloud.py` | Streamlit UI, authentication, model setup, and chat loop |
| `requirements.txt` | Dependencies installed locally and by Streamlit Cloud |
| `.streamlit/config.toml` | Theme and telemetry settings |
| `.streamlit/secrets.toml.example` | Placeholder OpenAI API key |
| `test_chatbot.py` | Credential-free tests with a mocked model |

## 1. OpenAI Prerequisites

- Create an API key in your [OpenAI project](https://platform.openai.com/api-keys).
- Configure API billing and confirm the key's project has access to your model.
   A ChatGPT subscription does not include OpenAI API usage.
- Confirm that `gpt-6-luna` is available and supports the requested parameters.

The app preserves your requested configuration:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
   "gpt-6-luna",
   model_provider="openai",
    temperature=0,
   top_p=1.0,
    seed=1234,
   reasoning_effort="none",
)
```

The actual app also supplies the API key, two retries, and a 60-second request
timeout. The requested model ID and parameter compatibility have not been
verified with a live OpenAI call. A 404 can mean an unknown model or missing
access; a 400 can mean unsupported parameters. No different model is silently
substituted. `top_p=1.0` is not equivalent to `top_k=1`, and temperature/seed do
not guarantee identical replies. Dependencies have upper bounds, not an exact
reproducibility lockfile.

## 2. Local Setup

Use Python **3.11 or 3.12** (3.11+ is required). From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Put Your API Key in Local Secrets (Recommended)

1. If `.streamlit/secrets.toml` does not exist, create it using
   `.streamlit/secrets.toml.example` as the template.
2. Set this single line to your real API key:

   ```toml
   OPENAI_API_KEY = "YOUR_REAL_OPENAI_API_KEY"
   ```

3. Keep this at the top level, not underneath `[gcp_service_account]` or another
   TOML section. The old Google credentials section is no longer used and can be
   removed from this app's secrets file. Do not delete credentials needed by
   other projects.
4. Never put the real key in the example template, Python source, or README.
   `.streamlit/secrets.toml` is ignored by Git; verify that before pushing:

   ```bash
   git check-ignore .streamlit/secrets.toml
   ```

### Alternative: Environment Variable

Set `OPENAI_API_KEY` in the same terminal before launching the app. On macOS,
this reads the key without displaying it or putting it into shell history:

```bash
read -r -s -p "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
```

The app prefers `OPENAI_API_KEY` in Streamlit secrets over the environment
variable when both exist. A `.env` file is not loaded automatically.
`GOOGLE_APPLICATION_CREDENTIALS` and service-account JSON are no longer used.

### Start the App

```bash
python -m streamlit run chatbot_cloud.py
```

Open http://localhost:8501. Use Streamlit to launch it, not
`python chatbot_cloud.py`: Streamlit supplies the browser UI and session state.

## 3. How It Works

1. Streamlit runs the script top to bottom. UI actions trigger reruns.
2. `st.session_state.messages` keeps role/content dictionaries for the current
   browser session. It survives reruns, but is not durable storage.
3. `get_model()` reads the API key and uses `init_chat_model` to create the OpenAI client.
   `st.cache_resource` reuses that client across reruns and sessions; it never
   caches conversation history or responses. Restart after changing credentials.
4. `build_messages()` creates a `SystemMessage`, converts saved turns into
   `HumanMessage` / `AIMessage` objects, and appends the new question.
5. `model.invoke(messages)` sends the conversation to OpenAI. The spinner
   stays visible until the complete response arrives; this demo does not stream.
6. `response.text` extracts answer text. Only successful, nonempty replies are
   saved as a complete user/assistant pair. A failed question can be submitted again.
7. The script reruns to redraw the chat and refresh the download button.

Only the last **20 saved messages** (10 complete turns), plus the system prompt
and new question, are sent to the model. Older messages remain visible and
downloadable but are outside the model's context. Inputs are limited to 4,000
characters. To experiment, edit `SYSTEM_PROMPT` or `HISTORY_LIMIT`; keep the
history limit even so complete turns are retained.

## 4. Deploy to Streamlit Community Cloud

### Prepare GitHub

This project may be a subdirectory of a larger Git repository. From this project
directory, inspect and stage only its files so unrelated changes are not included:

```bash
git status --short --untracked-files=all -- .
git add -- .gitignore .github/copilot-instructions.md .streamlit/config.toml .streamlit/secrets.toml.example README.md chatbot_cloud.py requirements.txt test_chatbot.py
git diff --cached --name-only
```

Review the staged names and contents before committing and pushing. If unrelated
files were already staged, handle those separately before making this commit.
Ignore rules do not protect credentials already tracked by Git.

### Prepare Cloud Secrets

Cloud does not inherit your local environment or ignored secrets file. In its
Secrets editor, supply the same single top-level entry with your real key:

```toml
OPENAI_API_KEY = "YOUR_REAL_OPENAI_API_KEY"
```

No service-account JSON, private-key conversion, or credential file path is
needed. Keep the key private and never paste it into chat or commit it to GitHub.

### Deploy

1. Push this project to a GitHub repository. Commit the app, requirements, config,
   README, and placeholder template, **never real credentials**. The local secrets
   file and common credential filenames are ignored; this is not a substitute
   for checking what you commit. Revoke any accidentally published key.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and create
   an app from that repository and branch.
3. Set the main file path to `chatbot_cloud.py` (include any subdirectory if needed).
   For a larger repository, run `git rev-parse --show-prefix` from this directory
   and append `chatbot_cloud.py` to the returned prefix for the main file path.
4. In **Advanced settings**, choose Python **3.11 or 3.12**. Paste the populated
   `OPENAI_API_KEY` entry into **Secrets**. Cloud installs `requirements.txt`.
5. Deploy, open the app URL, and submit a test question. Later, edit secrets through
   the app settings and reboot the app to rebuild the cached client.

For an existing Cloud app, update **Settings -> Secrets**, remove the old Google
section if it is no longer needed, save, and reboot. Push the updated dependency
file as well as the app. Confirm a real reply and a follow-up answer after deployment.

**Costs and privacy:** Streamlit hosting and OpenAI API billing are separate.
All users call OpenAI using the app's key, billed to its OpenAI project.
This learning demo has no app-level authentication or rate limiting.
Keep deployment access restricted, set quotas and billing alerts (alerts are not
hard spending caps), and do not expose an unrestricted public demo with production
credentials. Prompts and recent history are sent to OpenAI. Do not enter
sensitive data. New chat clears only this app's session history, not provider logs.

## 5. Tests

After installing dependencies:

```bash
python -m unittest -v test_chatbot
```

Tests use Streamlit's `AppTest`, fake API keys, and a mocked LangChain model:
no real credentials, API calls, or charges. They check key precedence, missing
keys, exact model settings, history bounds and roles, follow-up context, reset,
and failure handling. They do not prove live model availability or account access.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing API key | Top-level `OPENAI_API_KEY` in secrets or the terminal environment |
| Authentication failed / 401 | Correct, active OpenAI API key; restart after replacing it |
| Access denied / 403 | Key permissions and OpenAI project model access |
| Model/request unavailable / 404 or 400 | Exact model ID and support for `temperature`, `top_p`, `seed`, and `reasoning_effort` |
| Rate limit or quota exceeded / 429 | API billing, available credit, and request rate |
| Generic request failure | Secrets formatting, dependencies, network restrictions, or timeout |
| Missing imports | Activate the venv and install `requirements.txt`; select that interpreter in VS Code |
| Secrets changes not applied | Restart locally or reboot the Cloud app |

Raw exception details are not shown in the browser to avoid exposing credential
or infrastructure details. Never paste an API key into chat or support logs.

## References

- [Streamlit chat tutorial](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
- [Streamlit secrets management](https://docs.streamlit.io/develop/concepts/connections/secrets-management)
- [Streamlit Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [LangChain OpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai)