import os
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load API key from .env or environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    """Extract text from PDF."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def summarize(text, prompt_intro):
    """Summarize one chunk of text via GPT."""
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"{prompt_intro}\n\n```{text}```"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    return resp.choices[0].message.content.strip()

def chunk_text(text, max_chars=15000):
    """Split large text into chunks."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    parsed_result = None

    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            try:
                spec_text = extract_text(spec_file)
                subm_text = extract_text(subm_file)

                # Summarize spec in chunks
                chunks = chunk_text(spec_text)
                bullet_summaries = [
                    summarize(chunk, f"Chunk {i+1}: Summarize into 5–8 bullets")
                    for i, chunk in enumerate(chunks)
                ]
                merged_summary = "\n".join(bullet_summaries)
                spec_summary = summarize(merged_summary, "Summarize all bullets into 8 final bullet points:")

                # Compare with submittal
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a construction specifications expert. Compare the SUBMITTAL to the SPEC SUMMARY. "
                            "Return a JSON list with keys: requirement, provided, compliance (true/false), comment."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"SPEC SUMMARY:\n{spec_summary}\n\nFULL SUBMITTAL:\n{subm_text}"
                    }
                ]

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                result = response.choices[0].message.content.strip()

                # Try parsing the result as JSON
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = None

            except Exception as e:
                result = f"Error processing files or calling OpenAI: {e}"
                parsed_result = None

    # Debug log for Render
    print("\n=== GPT Result Start ===")
    print(result)
    print("=== Parsed:", "✅ success" if parsed_result else "❌ failed")
    print("=== GPT Result End ===\n")

    return render_template('index.html', result=result, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
