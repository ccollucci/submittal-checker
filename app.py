import os
import json
import time
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, max_chars=10000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def compare_spec_chunk_to_submittal(spec_chunk, subm_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a construction compliance analyst. Compare the SUBMITTAL to the SPEC CHUNK. "
                "First, provide a short summary paragraph. Then output a JSON array of issues using keys: "
                "requirement, provided, compliance (true/false), comment. Use double quotes, no markdown."
            )
        },
        {
            "role": "user",
            "content": f"SPEC CHUNK:\n{spec_chunk}\n\nSUBMITTAL:\n{subm_text}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()

def extract_json(text):
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else None

@app.route('/', methods=['GET', 'POST'])
def index():
    summary_blocks = []
    issues = []
    raw_responses = []

    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            try:
                spec_text = extract_text(spec_file)
                subm_text = extract_text(subm_file)
                spec_chunks = chunk_text(spec_text)

                for i, chunk in enumerate(spec_chunks):
                    print(f"Processing chunk {i+1}/{len(spec_chunks)}...")
                    chunk_result = compare_spec_chunk_to_submittal(chunk, subm_text)
                    raw_responses.append(chunk_result)

                    # Try splitting summary and JSON
                    summary_part = chunk_result.split("[", 1)[0].strip()
                    summary_blocks.append(summary_part)

                    json_part = extract_json(chunk_result)
                    if json_part:
                        try:
                            parsed = json.loads(json_part)
                            issues.extend(parsed)
                        except Exception as e:
                            print(f"❌ JSON parse failed for chunk {i+1}:", e)

                    time.sleep(1)  # throttle to avoid TPM limit

            except Exception as e:
                summary_blocks = [f"⚠️ Error during processing: {e}"]

    combined_summary = "\n\n".join(summary_blocks)
    return render_template('index.html', summary=combined_summary, parsed_result=issues, loading=False)

if __name__ == '__main__':
    app.run(debug=True)
