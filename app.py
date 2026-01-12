import os
import pickle
from typing import TypedDict, List, Optional
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing. Please check your .env file.")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = Groq(api_key=GROQ_API_KEY)

try:
    faiss_index = faiss.read_index("vector.index")
    documents = pickle.load(open("documents.pkl", "rb"))
except Exception as e:
    raise RuntimeError("Could not load index/documents. Did you run ingest.py?") from e


class AgentState(TypedDict):
    query: str
    session_id: str
    context: Optional[str]
    sources: List[str]
    is_relevant: bool
    answer: Optional[str]

SESSION_STORE = {}

def get_session_history(session_id: str):
    return SESSION_STORE.setdefault(session_id, [])

def add_message(session_id: str, role: str, content: str):
    SESSION_STORE.setdefault(session_id, []).append({
        "role": role,
        "content": content
    })

def set_pending_summarize(session_id: str, value: bool):
    SESSION_STORE.setdefault(session_id, []).append({
        "role": "system",
        "pending_summarize": value
    })

def has_pending_summarize(session_id: str) -> bool:
    history = SESSION_STORE.get(session_id, [])
    for msg in reversed(history):
        if "pending_summarize" in msg:
            return msg["pending_summarize"]
    return False

def store_last_context(session_id: str, context: str):
    SESSION_STORE.setdefault(session_id, []).append({
        "role": "system",
        "last_context": context
    })

def get_last_context(session_id: str):
    history = SESSION_STORE.get(session_id, [])
    for msg in reversed(history):
        if "last_context" in msg:
            return msg["last_context"]
    return None

SYSTEM_RAG_PROMPT = """
You are an AI assistant answering questions about internal company policies.

Rules:
1. Use ONLY the provided context.
2. If the answer is not in the context, say:
   "The information is not available in the internal documents."
3. Do not use external knowledge.
4. Keep the response clear, concise, and professional.
"""

SYSTEM_LLM_PROMPT = """
You are a helpful AI assistant.
Answer clearly and concisely.
If the question is ambiguous, ask for clarification.
"""

def is_yes_no(text: str):
    text = text.lower().strip()
    if text in ["yes", "y", "yes please", "ok", "summarize", "sure"]:
        return True
    if text in ["no", "n", "no thanks", "don't summarize", "nope"]:
        return False
    return None

CLASSIFIER_PROMPT = """
You are a query classifier. Determine if the user's question is about internal company policies.

Our internal documents cover:
- Employee FAQ
- Leave policies
- Security policies  
- Work from home policies

Respond with ONLY one word:
- "YES" if the query is about company policies, HR, leave, security, work policies, or internal procedures
- "NO" if it's a general knowledge question unrelated to company policies

Examples:
- "What is the leave policy?" → YES
- "How do I apply for leave?" → YES
- "What is the capital of India?" → NO
- "Tell me about security policies" → YES
- "What's the weather today?" → NO
"""

def classify_query(state: AgentState) -> AgentState:
    query = state["query"]
    
    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": query}
    ]
    
    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    
    classification = response.choices[0].message.content.strip().upper()
    state["is_relevant"] = "YES" in classification
    
    return state

def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]

    query_emb = embedder.encode([query], normalize_embeddings=True)
    scores, indices = faiss_index.search(query_emb, 3)

    texts, sources = [], []
    
    for idx in indices[0]:
        if idx < len(documents) and idx >= 0:
            doc = documents[idx]
            texts.append(doc["text"])
            sources.append(doc["source"])

    state["context"] = "\n\n".join(texts)
    state["sources"] = list(set(sources))

    return state


def llm_with_context(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    history = get_session_history(session_id)
    valid_history = [msg for msg in history if msg.get("role") in ["user", "assistant"]]

    messages = [
        {"role": "system", "content": SYSTEM_RAG_PROMPT},
        {"role": "system", "content": f"Context:\n{state['context']}"}
    ] + valid_history + [
        {"role": "user", "content": state["query"]}
    ]

    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )

    state["answer"] = response.choices[0].message.content

    add_message(session_id, "user", state["query"])
    add_message(session_id, "assistant", state["answer"])

    return state

def llm_only(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    history = get_session_history(session_id)
    valid_history = [msg for msg in history if msg.get("role") in ["user", "assistant"]]

    messages = [
        {"role": "system", "content": SYSTEM_LLM_PROMPT}
    ] + valid_history + [
        {"role": "user", "content": state["query"]}
    ]

    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )

    state["answer"] = response.choices[0].message.content
    state["sources"] = [] 

    add_message(session_id, "user", state["query"])
    add_message(session_id, "assistant", state["answer"])

    return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rag_llm", llm_with_context)
    graph.add_node("llm_only", llm_only)

    graph.set_entry_point("classify")

    def route_after_classify(state):
        return "retrieve" if state["is_relevant"] else "llm_only"

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "retrieve": "retrieve",
            "llm_only": "llm_only"
        }
    )

    graph.add_edge("retrieve", "rag_llm")
    graph.add_edge("rag_llm", END)
    graph.add_edge("llm_only", END)

    return graph.compile()

agent = build_graph()


app = FastAPI(title="LangGraph RAG Agent")

class AskRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/ask")
def ask(req: AskRequest):
    session_id = req.session_id
    user_input = req.query.strip()
    preference = is_yes_no(user_input)
    if preference is not None and has_pending_summarize(session_id):
        if preference:
            last_context = get_last_context(session_id)
            if last_context:
                summary_messages = [
                    {"role": "system", "content": "Summarize the following internal policy text concisely."},
                    {"role": "user", "content": last_context}
                ]
                response = llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=summary_messages
                )
                summary = response.choices[0].message.content
                
                set_pending_summarize(session_id, False)
                add_message(session_id, "user", user_input)
                add_message(session_id, "assistant", summary)
                
                return {"answer": summary, "sources": []}
    
        set_pending_summarize(session_id, False)
        add_message(session_id, "user", user_input)
        add_message(session_id, "assistant", "Okay.")
        return {"answer": "Okay.", "sources": []}

    initial_state = {
        "query": user_input,
        "session_id": session_id,
        "context": None,
        "sources": [],
        "is_relevant": False,
        "answer": None
    }

    result = agent.invoke(initial_state)
    response_text = result["answer"]
    if result["is_relevant"] and result.get("context"):
        store_last_context(session_id, result["context"])
        set_pending_summarize(session_id, True)
        response_text += "\n\nWould you like a summarized version? (yes/no)"

    return {
        "answer": response_text,
        "sources": result["sources"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)