import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# 1. Load PDF files
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("No readable text found in PDF")

    return full_text


# 2. Create chunks
def create_chunks(full_text, chunksize=500, overlap=100):

    chunks = []
    start = 0

    while start < len(full_text):

        end = start + chunksize

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. Build vectors
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("No chunks available for embedding")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(
        embed_vectors,
        dtype="float32"
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(
        embed_vectors.shape[1]
    )

    index.add(embed_vectors)

    return index, embed_model


# 4. Load medicine database
def load_base(pdf_path="medicine_docs"):

    chunks = []

    pdf_files = list(
        Path(pdf_path).glob("*.pdf")
    )

    if not pdf_files:
        raise ValueError("No medicine PDF files found")

    for pdf in pdf_files:

        full_text = load_pdf(pdf)

        chunks.extend(
            create_chunks(full_text)
        )

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# 5. Retrieve relevant content
def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=3
):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if not chunks:
        raise ValueError("Medicine database is empty")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed,
        dtype="float32"
    )

    faiss.normalize_L2(query_embed)

    scores, indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in indices[0]:

        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. Ask medicine RAG
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("API key missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    if not background.strip():
        return "No relevant medicine information found."

    prompt = """
    You are an educational Medicine Information Assistant.

    Answer using only the provided document background.

    Explain:
    1. Medicine uses
    2. How the medicine works
    3. Common side effects
    4. General precautions

    Safety rules:
    - Do not prescribe medicines.
    - Do not recommend personal dosages.
    - Do not diagnose diseases.
    - Do not advise users to start or stop medicines.
    - If information is missing, clearly say so.
    - Recommend consulting a doctor or pharmacist
      for personal medical advice.

    Document Background:
    """ + background + """

    User Question:
    """ + question + """

    Answer:
    """

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
