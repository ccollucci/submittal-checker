import os
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load OpenAI key from .env or environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

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

                # Build GPT prompt
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a construction specifications expert. "
                            "Compare the SUBMITTAL to the SPEC below. "
                            "Return a JSON list of issues using these keys: "
                            "requirement, provided, compliance (true/false), comment. "
                            "Only include fields that are verifiable and specific."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"SPEC:\n{spec_text}\n\nSUBMITTAL:\n{subm_text}"
                    }
                ]

                # Use GPT-4o (128k tokens)
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0
                )

                # Extract and clean result
                raw = response.choices[0].message.content
                result = raw.strip() if raw else "⚠️ GPT response was empty."

                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = None

            except Exception as e:
                result = f"⚠️ Error during comparison: {e}"
                parsed_result = None

    # Debug logging (visible in Render live logs)
    print("\n=== GPT Comparison Response ===")
    print(result if result else "⚠️ No response.")
    print("Parsed:", "✅" if parsed_result else "❌")
    print("===============================\n")

    return render_template('index.html', result=result, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
