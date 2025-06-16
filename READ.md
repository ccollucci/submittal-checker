# Submittal Checker

A Flask app that compares spec documents with submittals using OpenAI.

To deploy on Render:
1. Add your OpenAI API key as an environment variable.
2. Use `gunicorn app:app` as the start command.
3. Build with `pip install -r requirements.txt`.
