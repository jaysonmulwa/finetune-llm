import pdfplumber
import os
import re
import json
import requests

OLLAMA_URL = "" # path_to_local_ollama http://localhost:11434/api/generate
MODEL = ""   # change to your ollama model name (e.g. mistral, phi3, etc.)

def clean_ollama_response(text):
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end != -1:
        return text[start:end]
    return text


def ollama_generate(prompt):
    """
    Call Ollama API
    """
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "options": {"temperature": 0.7}
        },
        stream=True
    )
    output = ""
    for line in resp.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                output += data["response"]
    return output.strip()


def generate_examples(chunk, n_examples=3):
    prompt = f"""
    You are creating training data for fine-tuning a **customer-facing** chatbot.

    Here is a piece of company documentation:

    \"\"\"{chunk}\"\"\"

    Generate {n_examples} diverse Q&A pairs where the **user** is a customer asking about this documentation, 
    and the **assistant** provides a technically accurate, concise and context-aware answer. 
    Answers should be factual, strictly grounded in the provided text to explain the reasoning or implications clearly for a customer audience.
    The **assistant** should not make up answers or speculate beyond the provided text.

    Format the output as a strict JSON array of objects with this schema:
    [
    {{
        "user": "customer question here",
        "assistant": "long, technical answer here"
    }},
    ...
    ]
    Do not include anything else outside the JSON.
    """

    response_text = ollama_generate(prompt)
    # Try parsing into JSON
    try:
        return json.loads(clean_ollama_response(response_text))
    except Exception:
        print("⚠️ Could not parse Ollama response, got:\n", clean_ollama_response(response_text))
        return []


def generate_dataset(chunks):
    # ---- Generate dataset ----
    dataset = []
    for chunk in chunks:
        print(f"generating for {chunk}")
        examples = generate_examples(chunk, n_examples=3)
        print(f"examples: {len(examples)} {examples}")
        for ex in examples:
            dataset.append({
                "messages": [
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]}
                ]
            })
    
    print(f"dataset: {len(dataset)} {dataset}")

    # ---- Save to JSONL for fine-tuning ----
    with open("synthetic_training_data.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_pdf():
        """
        Parse a PDF file and extract text.
        This is a placeholder for any PDF parsing logic you might want to implement.
        """
        folder_path = "./path/to/pdf_files"

        # Dictionary to hold filename -> list of lines
        all_text_chunks = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing: {file_path}")
                
                full_text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                
                # Split into meaningful chunks (paragraphs or grouped sentences)
                chunks = re.split(r'\n{2,}|\.\s*\n', full_text)
                chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

                print(filename, chunks)
                
                all_text_chunks.extend(chunks)  # Add to the single array

        return all_text_chunks  # Return the list of text chunks for further processing


def parse_markdown():
    """
    Parse Markdown (.md) files in a folder and extract text chunks.
    Splits into paragraphs or long sentences for further processing.
    """
    folder_path = "./path/to/md_files"

    all_text_chunks = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # Split into meaningful chunks (paragraphs, headers, etc.)
            chunks = re.split(r'\n{2,}|\.\s*\n', full_text)
            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

            print(filename, f" → {len(chunks)} chunks")

            all_text_chunks.extend(chunks)

    return all_text_chunks

docs = parse_pdf()
generate_dataset(docs)
docs = parse_markdown()
generate_dataset(docs)