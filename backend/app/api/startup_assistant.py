from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationSummaryMemory
import numpy as np

load_dotenv()

router = APIRouter()

# Models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class StartupRequest(BaseModel):
    idea: str
    history: Optional[List[ChatMessage]] = None

class PlanResponse(BaseModel):
    tips: List[str]
    short_term: List[str]
    medium_term: List[str]
    long_term: List[str]
    cost_estimate: str
    time_estimate: str
    employee_suggestion: str
    additional_info: Optional[Dict[str, Any]] = None
    chat: List[ChatMessage]

# Embedding and vector store setup
_VECTORSTORE = None
_EMBEDDING_FUNCTION = None


def get_embedding_function():
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is None:
        _EMBEDDING_FUNCTION = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return _EMBEDDING_FUNCTION


def get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embedding = get_embedding_function()
        _VECTORSTORE = FAISS.load_local("backend/vector_store_startup", embedding, allow_dangerous_deserialization=True)
    return _VECTORSTORE


def calculate_relevance_scores(query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    dot_products = np.dot(doc_embeddings, query_embedding)
    similarities = dot_products / (doc_norms * query_norm)
    return similarities.tolist()


def retrieve_context(query: str, k: int = 5, score_threshold: float = 0.75) -> str:
    vectorstore = get_vectorstore()
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query)
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    if not results_with_scores:
        return "No relevant context found for this specific question."
    docs = [doc for doc, _ in results_with_scores]
    doc_embeddings = embedding_function.embed_documents([doc.page_content for doc in docs])
    relevance_scores = calculate_relevance_scores(query_embedding, doc_embeddings)
    scored_results = list(zip(docs, relevance_scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    filtered_results = [(doc, score) for doc, score in scored_results if score >= score_threshold]
    if not filtered_results:
        return "No highly relevant context found for this specific question."
    context_parts = []
    for doc, score in filtered_results:
        relevance_indicator = "High" if score > 0.85 else "Medium" if score > 0.75 else "Low"
        context_parts.append(f"[Relevance: {relevance_indicator}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)

# Memory for conversation
llm_for_memory = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
startup_memory = ConversationSummaryMemory(llm=llm_for_memory)

# Main endpoint
@router.post("/startup", response_model=PlanResponse)
async def startup_assistant(request: StartupRequest):
    user_message = request.idea
    chat_history = request.history or []
    # Get chat summary
    chat_summary = startup_memory.load_memory_variables({})["history"]
    # Retrieve context
    context = retrieve_context(user_message)
    # Build prompt
    system_prompt = (
        "You are a world-class startup advisor AI. "
        "You help users turn their ideas into actionable plans, including tips, timelines, cost, team, and more. "
        "Use the following context if relevant. Be practical, realistic, and encouraging."
    )
    user_prompt = f"""
    Startup Idea: {user_message}
    
    Context:
    {context}
    
    Conversation Summary:
    {chat_summary}
    """
    # Get response from LLM (OpenAI or Anthropic)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(system_prompt + "\n" + user_prompt)
    # Save to memory
    startup_memory.save_context({"input": user_prompt}, {"output": response})
    # Mock plan extraction (replace with actual parsing if needed)
    plan = PlanResponse(
        tips=["Validate your idea with real users.", "Start with a simple MVP."],
        short_term=["Market research", "Build MVP"],
        medium_term=["Launch beta", "Gather feedback"],
        long_term=["Scale operations", "Expand team"],
        cost_estimate="$5,000 - $20,000 for MVP phase",
        time_estimate="3-6 months for MVP",
        employee_suggestion="1-3 people for MVP phase",
        additional_info={"note": "This is a mock response. AI integration coming soon."},
        chat=chat_history + [ChatMessage(role="assistant", content=str(response))]
    )
    return plan 