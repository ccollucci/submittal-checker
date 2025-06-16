import os
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def extract_text(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def summarize(text, prompt_intro):
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"{prompt_intro}\n\n```{text}```"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    return resp.choices[0].message.content

def chunk_text(text, max_chars=15000):
    return [ text[i:i+max_chars] for i in range(0, len(text), max_chars) ]

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        spec_file = request.files.get('spec')
        subm_file = request.files.get('submittal')
        if spec_file and subm_file:
            spec_text = extract_text(spec_file)
            subm_text = extract_text(subm_file)

            chunks = chunk_text(spec_text)
            bullet_summaries = []
            for idx, chunk in enumerate(chunks, start=1):
                bullet_summaries.append(
                    summarize(
                        chunk,
                        prompt_intro=f"Chunk {idx}: Summarize into 5–8 bullets"
                    )
                )

            merged = "\n".join(bullet_summaries)
            spec_summary = summarize(
                merged,
                prompt_intro="Distill these bullets into 8 concise bullets:"
            )

            messages = [
                {"role":"system","content":(
                    "You are a construction specifications expert. "
                    "Compare the SUBMITTAL to the SPEC SUMMARY below. "
                    "Return JSON items with requirement, provided, compliance (true/false), comment."
                )},
                {"role":"user","content": f"SPEC SUMMARY:\n{spec_summary}\n\nFULL SUBMITTAL:\n{subm_text}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            result = response.choices[0].message.content

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
