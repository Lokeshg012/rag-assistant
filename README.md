# RAG-Powered Company Policy Assistant

An intelligent Retrieval-Augmented Generation (RAG) system for querying company policies using LangGraph, FastAPI, GROQ LLM, and FAISS vector search.

---

## 📋 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
  - [Local Setup](#local-setup)
  - [Azure Deployment](#azure-deployment)
- [Design Decisions](#design-decisions)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## 🏗️ Architecture Overview

### System Flow

```
User Query → FastAPI → LangGraph Agent → Response
                            ↓
                    [Classify Query]
                       /        \
                  Relevant?      Not Relevant
                     ↓               ↓
              [Retrieve Docs]   [LLM Only]
                     ↓               ↓
                 [RAG LLM]      [General Response]
                     ↓               ↓
              Answer + Sources  Answer (No Sources)
```

### Component Architecture

**1. Ingestion Pipeline (`rag_ingestion.py`)**
- Reads `.txt` files from `documents/` folder
- Chunks text (400 chars, 60 char overlap)
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores in FAISS index for fast retrieval

**2. Query Classification**
- LLM classifies if query is about company policies
- Routes to RAG pipeline or general LLM response

**3. RAG Pipeline**
- Retrieves top 3 relevant chunks from FAISS
- Provides context to GROQ LLM (Llama 3.3 70B)
- Generates contextual answers with source attribution

**4. Session Management**
- Maintains conversation history per session
- Offers summarization of retrieved policy documents
- Handles follow-up questions in context

**5. LangGraph State Machine**
- Orchestrates multi-step agent workflow
- Conditional routing based on query classification
- Stateful conversation management

---

## 🛠️ Tech Stack

### Core Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | REST API endpoints |
| **LLM** | GROQ - Llama 3.3 70B (Azure OpenAI was paid and I don't have one)  | Natural language understanding & generation |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Convert text to semantic vectors |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Fast similarity search |
| **Agent Framework** | LangGraph | Stateful agent orchestration |
| **Runtime** | Python 3.8+ | Application runtime |

### Dependencies
```
fastapi - API framework
uvicorn - ASGI server
groq - GROQ API client
sentence-transformers - Embedding generation
faiss-cpu - Vector similarity search
python-dotenv - Environment management
langgraph - Agent workflow management
```

---

## 🚀 Setup Instructions

### Local Setup

#### Prerequisites
- Python 3.8 or higher
- GROQ API key ([Get it here](https://console.groq.com))

#### Installation Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Assignment
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your GROQ API key
GROQ_API_KEY=your_groq_api_key_here
```

5. **Prepare documents**
- Add your `.txt` policy documents to the `documents/` folder
- Sample documents are already included:
  - `employee_faq.txt`
  - `leave_policy.txt`
  - `security_policy.txt`
  - `work_from_home_policy.txt`

6. **Run ingestion to create vector index**
```bash
python rag_ingestion.py
```
This creates:
- `vector.index` - FAISS index file
- `documents.pkl` - Serialized document chunks

7. **Start the API server**
```bash
python app.py
```
Server runs at `http://localhost:8000`

8. **Test the API**
```bash
# Using curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?", "session_id": "user123"}'

# Using Python
import requests
response = requests.post("http://localhost:8000/ask", json={
    "query": "What is the leave policy?",
    "session_id": "user123"
})
print(response.json())
```

### Azure Deployment

#### Option 1: Azure App Service

1. **Prerequisites**
- Azure CLI installed
- Azure subscription

2. **Create App Service**
```bash
# Login to Azure
az login

# Create resource group
az group create --name rag-agent-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name rag-agent-plan \
  --resource-group rag-agent-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group rag-agent-rg \
  --plan rag-agent-plan \
  --name rag-agent-app \
  --runtime "PYTHON:3.9"
```

3. **Configure environment variables**
```bash
az webapp config appsettings set \
  --resource-group rag-agent-rg \
  --name rag-agent-app \
  --settings GROQ_API_KEY="your_key_here"
```

4. **Deploy application**
```bash
# Using ZIP deployment
zip -r deploy.zip . -x "*.git*" "venv/*" "__pycache__/*"
az webapp deployment source config-zip \
  --resource-group rag-agent-rg \
  --name rag-agent-app \
  --src deploy.zip
```

5. **Configure startup command**
```bash
az webapp config set \
  --resource-group rag-agent-rg \
  --name rag-agent-app \
  --startup-file "startup.sh"
```

**Update `startup.sh`:**
```bash
#!/bin/bash
python rag_ingestion.py
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

#### Option 2: Azure Container Instances

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python rag_ingestion.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and push to Azure Container Registry**
```bash
# Create ACR
az acr create --resource-group rag-agent-rg \
  --name ragagentacr --sku Basic

# Build image
az acr build --registry ragagentacr \
  --image rag-agent:latest .

# Deploy to ACI
az container create \
  --resource-group rag-agent-rg \
  --name rag-agent-container \
  --image ragagentacr.azurecr.io/rag-agent:latest \
  --dns-name-label rag-agent-demo \
  --ports 8000 \
  --environment-variables GROQ_API_KEY="your_key"
```

---

## 🎯 Design Decisions

### 1. **LangGraph for Agent Orchestration**
**Why:** Provides stateful workflow management with conditional routing, making it easier to handle complex multi-step agent logic compared to simple function chains.

### 2. **Query Classification First**
**Why:** Avoids unnecessary RAG retrieval for general queries. Saves API calls and improves response time for off-topic questions.

### 3. **FAISS Over Traditional Databases**
**Why:** FAISS provides sub-millisecond vector similarity search even with large document collections. Lightweight and doesn't require separate database infrastructure.

### 4. **Chunk Size: 400 chars with 60 char overlap**
**Why:** 
- 400 chars captures enough context for policy paragraphs
- 60 char overlap prevents information loss at chunk boundaries
- Balances retrieval precision vs context window limits

### 5. **Top-K = 3 Documents**
**Why:** Provides sufficient context without overwhelming the LLM. Tested empirically for balance between relevance and noise.

### 6. **Session-based Context Management**
**Why:** Enables multi-turn conversations while keeping each session isolated. In-memory storage is sufficient for demo; production would use Redis/DynamoDB.

### 7. **Summarization on Demand**
**Why:** Users might want detailed policy info or quick summaries. Interactive approach gives users control.

### 8. **Source Attribution**
**Why:** Transparency and verifiability are critical for policy questions. Users can trace answers back to source documents.

### 9. **Temperature Settings**
- **RAG mode: 0.0** - Deterministic responses strictly based on context
- **General mode: 0.7** - More creative for general conversation

### 10. **GROQ (Llama 3.3 70B) as LLM**
**Why:** 
- Fast inference speeds (important for user experience)
- Strong instruction-following for RAG tasks
- Cost-effective compared to GPT-4

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

1. **In-Memory Session Storage**
   - Sessions lost on restart
   - Not suitable for production at scale

2. **No Authentication/Authorization**
   - Open API without user management
   - No rate limiting

3. **Single Language Support**
   - Only handles English documents
   - No multilingual embedding support

4. **Document Format**
   - Only `.txt` files supported
   - No PDF, DOCX, or HTML parsing

5. **Basic Chunking Strategy**
   - Character-based splitting doesn't respect semantic boundaries
   - May split mid-sentence or mid-paragraph

6. **No Evaluation Metrics**
   - No automated testing of answer quality
   - No retrieval accuracy tracking

7. **Limited Scalability**
   - FAISS in-memory index
   - Single server deployment

8. **No Document Updates**
   - Requires full re-ingestion for document changes
   - No incremental index updates

### Future Improvements

#### Short-term (1-2 weeks)
- [ ] Add Redis for session persistence
- [ ] Implement API authentication (JWT tokens)
- [ ] Add rate limiting middleware
- [ ] Support PDF document ingestion
- [ ] Add logging and monitoring (structured logs)
- [ ] Implement answer feedback mechanism (thumbs up/down)

#### Medium-term (1-2 months)
- [ ] Semantic chunking (respecting paragraph boundaries)
- [ ] Hybrid search (BM25 + vector search)
- [ ] Multi-language support with multilingual embeddings
- [ ] Document versioning and change tracking
- [ ] Admin dashboard for document management
- [ ] A/B testing framework for prompt variations
- [ ] Response caching for common queries
- [ ] Export conversation history

#### Long-term (3-6 months)
- [ ] Graph-based knowledge representation
- [ ] Auto-generated FAQ from documents
- [ ] Voice interface integration
- [ ] Fine-tuned embeddings for domain-specific terms
- [ ] Distributed FAISS with sharding
- [ ] Real-time document updates via CDC
- [ ] Advanced analytics dashboard
- [ ] Multi-modal support (images, tables in documents)
- [ ] Automated evaluation pipeline with ground truth dataset
- [ ] Integration with Slack/Teams for chatbot deployment

---

## 📂 Project Structure

```
Assignment/
├── app.py                    # Main FastAPI application
├── rag_ingestion.py          # Document ingestion pipeline
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .env.example              # Example environment file
├── .gitignore                # Git ignore rules
├── startup.sh                # Azure startup script
├── README.md                 # This file
├── documents/                # Policy documents
│   ├── employee_faq.txt
│   ├── leave_policy.txt
│   ├── security_policy.txt
│   └── work_from_home_policy.txt
├── vector.index              # FAISS index (generated)
└── documents.pkl             # Document chunks (generated)
```

---

## 📝 API Documentation

### POST `/ask`

**Request:**
```json
{
  "query": "What is the leave policy?",
  "session_id": "user123"
}
```

**Response:**
```json
{
  "answer": "The leave policy allows employees to take 20 days of paid leave per year...",
  "sources": ["leave_policy.txt"]
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is for educational/demo purposes.

---

## 🙋 Support

For issues or questions, please open a GitHub issue.
