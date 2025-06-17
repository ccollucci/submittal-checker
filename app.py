import os
import json
import time
import hashlib
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = ".cache"
CACHE_TTL = 86400  # 24 hours in seconds

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload files under 100MB.", 413

def extract_text(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cache_path(spec_hash):
    return os.path.join(CACHE_DIR, f"spec_{spec_hash}.json")

def load_cached_requirements(spec_text):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    spec_hash = hash_text(spec_text)
    cache_path = get_cache_path(spec_hash)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            if time.time() - data["timestamp"] < CACHE_TTL:
                print("✅ Using cached requirements")
                return data["requirements"]
            else:
                print("🕓 Cache expired for spec")
    return None

def save_requirements_to_cache(spec_text, requirements):
    spec_hash = hash_text(spec_text)
    cache_path = get_cache_path(spec_hash)
    data = {
        "timestamp": time.time(),
        "requirements": requirements
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)

def call_gpt_with_retry(prompt_messages, retries=3, delay=2.5):
    for attempt in range(retries):
        try:
            return openai.ChatCompletion.create(
                model="gpt-4o",
                messages=prompt_messages,
                temperature=0
            )
        except openai.error.RateLimitError:
            print(f"⚠️ Rate limit hit. Retry {attempt + 1}/{retries} in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    raise Exception("❌ Exceeded retry attempts due to rate limiting.")

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

                # Check cache
                requirements = load_cached_requirements(spec_text)
                if not requirements:
                    # Step 1: Extract requirements
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

                    extract_response = call_gpt_with_retry(extract_prompt)
                    time.sleep(2.5)
                    req_json = extract_response.choices[0].message.content.strip()
                    requirements = json.loads(req_json)

                    # Save to cache
                    save_requirements_to_cache(spec_text, requirements)

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

                compare_response = call_gpt_with_retry(compare_prompt)
                result_json = compare_response.choices[0].message.content.strip()
                parsed_result = json.loads(result_json)

                summary = "Comparison completed successfully."

            except Exception as e:
                summary = f"⚠️ Error: {e}"

    return render_template('index.html', summary=summary, parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
