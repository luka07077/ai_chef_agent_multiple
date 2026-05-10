import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from rag.vector_stores import search_knowledge
from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
Agentic RAG core module.
Implements a proper 3-step Agentic RAG pipeline:
  1. Query Rewriting   - reword the user's question to improve retrieval recall
  2. Multi-path Retrieval - search with both original + rewritten queries, deduplicate results
  3. Self-Reflection   - a lightweight judge model checks if the results are actually relevant
"""

logger = get_logger("ai_chef.agentic_rag")


def _get_evaluator_llm():
    """Get the lightweight judge model (keeps the main agent's compute free)."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    config = get_agent_config()
    return ChatTongyi(
        model_name=config["llm"]["evaluator_model"],
        dashscope_api_key=api_key,
        temperature=config["llm"]["evaluator_temperature"]
    )


def _rewrite_query(query: str, llm) -> list:
    """
    Step 1: Query Rewriting
    Reword the user's natural language question into better keywords for vector search.
    """
    prompts = get_prompt_config()
    rewrite_prompt = ChatPromptTemplate.from_template(prompts["rag_query_rewrite_prompt"])
    chain = rewrite_prompt | llm

    try:
        result = chain.invoke({"query": query})
        rewritten = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
        logger.info(f"[Query Rewriting] Original: '{query}' -> Rewritten: {rewritten}")
        return rewritten
    except Exception as e:
        logger.warning(f"[Query Rewriting] Rewrite failed, using original query: {e}")
        return []


def _multi_path_retrieve(query: str, rewritten_queries: list, k: int = 3) -> list:
    """
    Step 2: Multi-path Retrieval
    Search with both the original and rewritten queries, then combine and deduplicate.
    """
    all_docs = []
    seen_contents = set()

    # Search with original query
    for doc in search_knowledge(query, k=k):
        content_key = doc.page_content[:100]
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            all_docs.append(doc)

    # Search with each rewritten query
    for rq in rewritten_queries:
        for doc in search_knowledge(rq, k=k):
            content_key = doc.page_content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_docs.append(doc)

    logger.info(f"[Multi-path Retrieval] Retrieved {len(all_docs)} unique document chunks")
    return all_docs


def _self_reflect(query: str, docs: list, llm) -> str:
    """
    Step 3: Self-Reflection
    The judge model checks whether the retrieved docs actually answer the user's question.
    """
    raw_context = "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs]
    )

    prompts = get_prompt_config()
    reflection_prompt = ChatPromptTemplate.from_template(prompts["rag_reflection_prompt"])
    chain = reflection_prompt | llm

    result = chain.invoke({"query": query, "context": raw_context})
    return result.content.strip()


@tool
def search_private_knowledge(query: str) -> str:
    """
    Call this tool when the user asks about specific recipes, health tips, nutrition combos,
    or other questions that should be answered from the knowledge base.
    This tool runs the full Agentic RAG pipeline: query rewriting -> multi-path retrieval -> self-reflection.

    Args:
        query: the user's original question or search keywords.
    """
    logger.info(f"[Agentic RAG] Starting full retrieval pipeline, query: '{query}'")

    # Get the judge model (reuse same instance)
    evaluator_llm = _get_evaluator_llm()

    # Step 1: rewrite the query
    rewritten_queries = _rewrite_query(query, evaluator_llm)

    # Step 2: multi-path retrieval
    config = get_agent_config()
    k = config["rag"]["retrieval_top_k"]
    docs = _multi_path_retrieve(query, rewritten_queries, k=k)

    if not docs:
        logger.warning("[Agentic RAG] No relevant content found in knowledge base")
        return "Nothing relevant found in the local knowledge base. Please answer using general knowledge."

    # Step 3: self-reflection and filtering
    logger.info("[Agentic RAG] Running self-reflection check...")
    output = _self_reflect(query, docs, evaluator_llm)

    # Routing based on reflection result
    if "NOT_FOUND" in output:
        logger.info("[Agentic RAG] Reflection says: retrieved docs are unrelated, blocking.")
        return "Retrieved documents are not relevant to the question. Please answer using general knowledge."
    else:
        logger.info("[Agentic RAG] Reflection says: content is highly relevant, delivering.")
        return f"【High-quality info from knowledge base】\n{output}"


if __name__ == "__main__":
    print("=== Agentic RAG standalone test ===")
    test_q = "Based on my health report, can I eat seafood?"
    result = search_private_knowledge.invoke({"query": test_q})
    print(f"\nResult:\n{result}")
