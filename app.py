import os
import json
import time
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

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
    is_processing = False

    if request.method == 'POST':
        is_processing = True
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')

        if spec_file and subm_file:
            try:
                spec_text = extract_text(spec_file)
                subm_text = extract_text(subm_file)

                # Step 1: Extract requirements
                extract_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an architectural compliance assistant. Extract enforceable requirements from the provided specification."
                            " Return only a valid JSON array of requirement strings. No explanation. No markdown formatting."
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

                time.sleep(10)
                raw_json = extract_response.choices[0].message.content.strip()

                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                raw_json = raw_json.strip()

                print("GPT extracted requirements raw JSON:")
                print(raw_json)

                if not raw_json or not raw_json.startswith("["):
                    raise ValueError("GPT did not return valid JSON")

                requirements = json.loads(raw_json)

                # Step 2: Compare
                compare_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Compare each requirement to the submittal. For each, return an object with:"
                            " requirement, provided, compliance (true/false), and comment. Return only a JSON array of these objects."
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

                if result_json.startswith("```json"):
                    result_json = result_json[7:]
                if result_json.endswith("```"):
                    result_json = result_json[:-3]
                result_json = result_json.strip()

                print("GPT comparison result raw JSON:")
                print(result_json)

                if not result_json or not result_json.startswith("["):
                    raise ValueError("GPT did not return valid comparison JSON")

                parsed_result = json.loads(result_json)
                summary = "Comparison completed successfully."

            except Exception as e:
                summary = f"⚠️ Error: {e}"

        is_processing = False

    return render_template('index.html', summary=summary, parsed_result=parsed_result, is_processing=is_processing)

if __name__ == '__main__':
    app.run(debug=True)
