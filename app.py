import os
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB upload limit

# Handle 413 errors (file too large)
@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def summarize(text, prompt_intro):
    """Summarize a given block of text using GPT."""
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"{prompt_intro}\n\n```{text}```"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

def chunk_text(text, max_chars=15000):
    """Split large text into chunks."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    parsed = None

    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            spec_text = extract_text(spec_file)
            subm_text = extract_text(subm_file)

            # Summarize the spec in chunks
            chunks = chunk_text(spec_text)
            summaries = [summarize(chunk, f"Chunk {i+1}: Summarize in 5–8 bullets") for i, chunk in enumerate(chunks)]
            merged = "\n".join(summaries)
            spec_summary = summarize(merged, "Distill these bullet summaries into 8 final bullets:")

            # Generate comparison using the summary and full submittal
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a construction specifications expert. Compare the SUBMITTAL to the SPEC SUMMARY below. "
                        "Return a JSON list of items with keys: requirement, provided, compliance (true/false), comment."
                    )
                },
                {
                    "role": "user",
                    "content": f"SPEC SUMMARY:\n{spec_summary}\n\nFULL SUBMITTAL:\n{subm_text}"
                }
            ]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                result = response.choices[0].message.content

                # Try to parse result into JSON format
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = None

            except Exception as e:
                result = f"Error calling OpenAI: {e}"

    return render_template('index.html', result=result, parsed_result=parsed)

if __name__ == '__main__':
    app.run(debug=True)
