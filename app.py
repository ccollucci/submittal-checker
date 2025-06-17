import os
import json
import re
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load OpenAI key
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

def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else None

@app.route('/', methods=['GET', 'POST'])
def index():
    summary_text = None
    result = None
    parsed_result = None

    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            try:
                spec_text = extract_text(spec_file)
                subm_text = extract_text(subm_file)

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a construction compliance analyst."
                            " Compare the SUBMITTAL to the SPEC."
                            " First, provide a brief summary of the overall compliance in 3–5 sentences."
                            " Then below that, return a JSON array with keys: requirement, provided, compliance (true/false), comment."
                            " Use double quotes and do not wrap the JSON in code blocks or markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"SPEC:\n{spec_text}\n\nSUBMITTAL:\n{subm_text}"
                    }
                ]

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0
                )

                raw = response.choices[0].message.content
                if raw:
                    result = raw.strip()

                    # Try to split summary and JSON
                    summary_text = result.split("[", 1)[0].strip()
                    json_block = extract_json(result)

                    if json_block:
                        parsed_result = json.loads(json_block)
                else:
                    result = "⚠️ GPT response was empty."

            except Exception as e:
                result = f"⚠️ Error: {e}"

    print("\n=== GPT Compliance Output ===")
    print(result)
    print("Parsed JSON:", "✅" if parsed_result else "❌")
    print("==============================\n")

    return render_template('index.html', summary=summary_text, result=result, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
