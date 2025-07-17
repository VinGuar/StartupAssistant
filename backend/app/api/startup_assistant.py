from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationSummaryMemory
import numpy as np
import json

load_dotenv()

router = APIRouter()

# Models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class StartupRequest(BaseModel):
    fields: Optional[Dict[str, str]] = None
    history: Optional[List[ChatMessage]] = None

class PlanResponse(BaseModel):
    tips: List[str]
    cost_estimate: str
    time_estimate: str
    employee_suggestion: str
    short_term: List[str]
    medium_term: List[str]
    long_term: List[str]
    additional_info: Optional[Dict[str, Any]] = None
    chat: List[ChatMessage]
    follow_up_question: Optional[str] = None  # New field for follow-up questions
    plans: Optional[List[Dict[str, Any]]] = None  # New field for multiple plans

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
    fields = request.fields or {}
    user_message = fields.get('idea', '')
    chat_history = request.history or []
    # Check if the idea is too vague (e.g., very short or generic)
    if not user_message or len(user_message.strip()) < 10:
        follow_up = "Can you provide more details about your startup idea? For example, who is your target audience, what problem are you solving, and what makes your idea unique?"
        plan = PlanResponse(
            tips=[],
            cost_estimate="",
            time_estimate="",
            employee_suggestion="",
            short_term=[],
            medium_term=[],
            long_term=[],
            additional_info={"note": "Please provide more information to get a personalized plan."},
            chat=chat_history + [ChatMessage(role="assistant", content=follow_up)],
            follow_up_question=follow_up
        )
        return plan
    # Get chat summary
    chat_summary = startup_memory.load_memory_variables({})["history"]
    # Retrieve context
    context = retrieve_context(user_message)
    # Build prompt
    # Add all fields to the prompt
    fields_str = "\n".join([f"{k}: {v}" for k, v in fields.items()])
    system_prompt = (
        "You are a world-class startup advisor AI. "
        "You help users turn their ideas into actionable plans, including tips, timelines, cost, team, and more. "
        "Use the following context if relevant. Be practical, realistic, and encouraging. "
        "If the user's idea is missing key details (such as target audience, problem, or unique value), ask clarifying questions before giving a plan. "
        "ALWAYS return your response as a JSON object with the following fields: "
        "plans (list of 2-3 plan objects, each with: tips (list of strings), cost_estimate (string), time_estimate (string), employee_suggestion (string), short_term (list of strings), medium_term (list of strings), long_term (list of strings), additional_info (object, optional)). "
        "Example: {\"plans\": [{\"tips\": [\"tip1\", \"tip2\"], \"cost_estimate\": \"...\", \"time_estimate\": \"...\", \"employee_suggestion\": \"...\", \"short_term\": [\"...\"], \"medium_term\": [\"...\"], \"long_term\": [\"...\"], \"additional_info\": {\"note\": \"...\"}}, ...]}. "
        "If you need more information, ask the user specific questions instead of giving a plan."
    )
    user_prompt = f"""
    Startup Info:
    {fields_str}
    
    Context:
    {context}
    
    Conversation Summary:
    {chat_summary}
    """
    # Get response from LLM (OpenAI or Anthropic)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(system_prompt + "\n" + user_prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    # Check if the LLM is asking for more info (simple heuristic)
    follow_up = None
    if any(q in response_text.lower() for q in ["can you provide", "please specify", "what is your target audience", "what problem are you solving", "could you clarify"]):
        follow_up = response_text
    # Save to memory
    startup_memory.save_context({"input": user_prompt}, {"output": response_text})
    # Try to parse the response as JSON
    plan_data = None
    try:
        plan_data = json.loads(response_text)
    except Exception:
        plan_data = None
    if plan_data and not follow_up:
        # If multiple plans are present, return them
        if 'plans' in plan_data and isinstance(plan_data['plans'], list):
            # Use the first plan for the main fields, but include all in 'plans'
            first = plan_data['plans'][0] if plan_data['plans'] else {}
            plan = PlanResponse(
                tips=first.get("tips", []),
                cost_estimate=first.get("cost_estimate", ""),
                time_estimate=first.get("time_estimate", ""),
                employee_suggestion=first.get("employee_suggestion", ""),
                short_term=first.get("short_term", []),
                medium_term=first.get("medium_term", []),
                long_term=first.get("long_term", []),
                additional_info=first.get("additional_info", {}),
                chat=chat_history + [ChatMessage(role="assistant", content=response_text)],
                follow_up_question=None,
                plans=plan_data['plans']
            )
            return plan
        # Fallback to single plan if only one
        plan = PlanResponse(
            tips=plan_data.get("tips", []),
            cost_estimate=plan_data.get("cost_estimate", ""),
            time_estimate=plan_data.get("time_estimate", ""),
            employee_suggestion=plan_data.get("employee_suggestion", ""),
            short_term=plan_data.get("short_term", []),
            medium_term=plan_data.get("medium_term", []),
            long_term=plan_data.get("long_term", []),
            additional_info=plan_data.get("additional_info", {}),
            chat=chat_history + [ChatMessage(role="assistant", content=response_text)],
            follow_up_question=None,
            plans=None
        )
        return plan
    # Fallback to mock plan if parsing fails or follow-up is needed
    plan = PlanResponse(
        tips=["Validate your idea with real users.", "Start with a simple MVP."],
        cost_estimate="$5,000 - $20,000 for MVP phase",
        time_estimate="3-6 months for MVP",
        employee_suggestion="1-3 people for MVP phase",
        short_term=["Market research", "Build MVP"],
        medium_term=["Launch beta", "Gather feedback"],
        long_term=["Scale operations", "Expand team"],
        additional_info={"note": "This is a mock response. AI integration coming soon."},
        chat=chat_history + [ChatMessage(role="assistant", content=response_text)],
        follow_up_question=follow_up
    )
    return plan 