import os
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

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

                extract_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Extract enforceable requirements from the specification. Return only a JSON array of strings."},
                        {"role": "user", "content": f"{spec_text}"}
                    ],
                    temperature=0
                )

                raw_json = extract_response.choices[0].message.content.strip()
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                requirements = json.loads(raw_json)

                compare_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Compare each requirement to the submittal. Return a JSON array with requirement, provided, compliance (true/false), and comment."},
                        {"role": "user", "content": f"REQUIREMENTS:\n{json.dumps(requirements)}\n\nSUBMITTAL:\n{subm_text}"}
                    ],
                    temperature=0
                )

                result_json = compare_response.choices[0].message.content.strip()
                if result_json.startswith("```json"):
                    result_json = result_json[7:]
                if result_json.endswith("```"):
                    result_json = result_json[:-3]
                parsed_result = json.loads(result_json)
                summary = "Comparison completed successfully."

            except Exception as e:
                summary = f"Error: {e}"

    return render_template('index.html', summary=summary, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
