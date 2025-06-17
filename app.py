import os
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load API key from environment or .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def summarize(text, prompt_intro):
    """Summarize a block of text using OpenAI GPT."""
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
    return response.choices[0].message.content.strip()

def chunk_text(text, max_chars=15000):
    """Split large input into smaller text chunks."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

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

                # Step 1: Summarize spec in chunks
                chunks = chunk_text(spec_text)
                bullet_summaries = [
                    summarize(chunk, f"Chunk {i + 1}: Summarize in 5–8 bullets")
                    for i, chunk in enumerate(chunks)
                ]
                merged = "\n".join(bullet_summaries)

                # Step 2: Final spec summary
                spec_summary = summarize(
                    merged,
                    "Combine these bullets into 8 final summary points:"
                )

                # Step 3: Run comparison prompt
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a construction specifications expert. Compare the SUBMITTAL to the SPEC SUMMARY. "
                            "Return a JSON list of items with keys: requirement, provided, compliance (true/false), comment."
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

                # Step 4: Extract result from GPT response
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                else:
                    result = "⚠️ GPT did not return a valid response."

                # Step 5: Try parsing the JSON
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = None

            except Exception as e:
                result = f"Error processing or calling OpenAI: {e}"
                parsed_result = None

    # Debug logs
    print("\n=== RAW GPT RESPONSE ===")
    print(result)
    print("Parsed:", "✅" if parsed_result else "❌")
    print("========================\n")

    return render_template('index.html', result=result, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
