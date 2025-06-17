import os
import json
import time
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

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    parsed_result = []

    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            try:
                spec_text = extract_text(spec_file)
                subm_text = extract_text(subm_file)

                # Step 1: Extract only requirements from spec
                extract_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an architectural compliance assistant. Your task is to extract only enforceable, specific requirements from the provided specification."
                            " Do not summarize or paraphrase. List the requirements exactly as written, in their original wording. Return the list as a JSON array of strings."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"SPECIFICATION:\n{spec_text}"
                    }
                ]

                extract_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=extract_prompt,
                    temperature=0
                )

                time.sleep(2.5)  # Wait to stay within TPM limit

                req_json = extract_response.choices[0].message.content.strip()
                requirements = json.loads(req_json)

                # Step 2: Compare those requirements to the full submittal
                compare_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a construction compliance checker. Compare each requirement against the SUBMITTAL below."
                            " Return a JSON array of objects with: requirement, provided, compliance (true/false), comment."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"REQUIREMENTS:\n{json.dumps(requirements)}\n\nSUBMITTAL:\n{subm_text}"
                    }
                ]

                compare_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=compare_prompt,
                    temperature=0
                )

                result_json = compare_response.choices[0].message.content.strip()
                parsed_result = json.loads(result_json)

                summary = "Comparison completed successfully."

            except Exception as e:
                summary = f"⚠️ Error: {e}"

    return render_template('index.html', summary=summary, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
