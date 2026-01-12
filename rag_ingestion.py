import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DOC_FOLDER = "documents"
INDEX_PATH = "vector.index"
DOCS_PATH = "documents.pkl"
CHUNK_SIZE = 400 
OVERLAP = 60

if not os.path.exists(DOC_FOLDER):
    os.makedirs(DOC_FOLDER)
    print(f"Created folder '{DOC_FOLDER}'. Please add .txt files there and rerun.")
    exit()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - OVERLAP
    return chunks

documents = []
texts = []

files = [f for f in os.listdir(DOC_FOLDER) if f.endswith(".txt")]
if not files:
    print(f"No .txt files found in {DOC_FOLDER}")
    exit()

print(f"Processing {len(files)} files...")

for file in files:
    path = os.path.join(DOC_FOLDER, file)
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)

    for chunk in chunks:
        documents.append({
            "text": chunk,
            "source": file
        })
        texts.append(chunk)

print(f"Generated {len(texts)} text chunks.")

embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"Ingestion complete")