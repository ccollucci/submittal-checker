import os
import time
import markdown2
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

def ensure_valid_markdown_table(table_text):
    # Ensure table has a proper header and separator line
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) >= 2 and "|" in lines[0]:
        header = lines[0]
        separator = "|".join(["---"] * (header.count("|") - 1))
        lines.insert(1, separator)
    return "\n".join(lines)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    rendered_table = ""

    if request.method == 'POST':
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
                            "You are an architectural compliance assistant. Extract enforceable requirements from the specification."
                            " List each one briefly as a bullet point."
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
                requirements_text = extract_response.choices[0].message.content.strip()

                # Step 2: Compare requirements to submittal using markdown table and summary
                compare_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Compare each requirement to the submittal."
                            " First provide a brief summary (2-3 sentences) of the overall compliance."
                            " Then provide a markdown table with columns: Requirement | Provided | Compliant (Yes/No) | Comment."
                            " Make sure the markdown table includes a proper header separator using --- on the second line."
                            " Only return the summary paragraph followed by the table."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"REQUIREMENTS:\n{requirements_text}\n\nSUBMITTAL:\n{subm_text}"
                    }
                ]

                compare_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=compare_prompt,
                    temperature=0
                )

                full_output = compare_response.choices[0].message.content.strip()

                # Separate summary and table
                split = full_output.split("| Requirement |", 1)
                if len(split) == 2:
                    summary = split[0].strip()
                    raw_table = "| Requirement |" + split[1].strip()
                else:
                    summary = full_output
                    raw_table = ""

                raw_table = ensure_valid_markdown_table(raw_table)
                rendered_table = markdown2.markdown(raw_table)

            except Exception as e:
                summary = f"⚠️ Error: {e}"

    print("SUMMARY:")
    print(summary)
    print("RENDERED TABLE:")
    print(rendered_table)

    return render_template(
    'index.html',
    summary=summary or "",
    rendered_table=rendered_table or ""
)


if __name__ == '__main__':
    app.run(debug=True)
