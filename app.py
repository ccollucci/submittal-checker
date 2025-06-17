import os
import json
import time
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

def extract_json_array(text):
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    return match.group(0) if match else None

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

                # Basic input validation
                if len(spec_text) < 100 or len(subm_text) < 100:
                    raise ValueError("One or both PDFs may be empty or too short.")

                # Step 1: Extract requirements from spec
                extract_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an architectural compliance assistant. Your task is to extract only enforceable, specific requirements from the provided specification."
                            " Do not summarize or paraphrase. Return only a valid JSON array of strings. Do not include any comments, explanations, or formatting. Use proper escape characters and close all quotes."
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

                time.sleep(10)  # safer delay
                raw_output = extract_response.choices[0].message.content.strip()
                print("GPT Extracted Requirements Raw Output:")
                print(raw_output)

                clean_json = extract_json_array(raw_output)
                if not clean_json:
                    print("⚠️ GPT returned non-JSON:", raw_output)
                    raise ValueError("GPT did not return valid JSON.")

                try:
                    requirements = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    print("❌ Failed to parse JSON:")
                    print(clean_json)
                    raise ValueError(f"Invalid JSON format: {e}")

                # Step 2: Compare requirements to submittal
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
