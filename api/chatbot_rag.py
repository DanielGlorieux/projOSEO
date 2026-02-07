"""
Chatbot RAG intelligent pour agents de l'eau ONEA
Basé sur le système de documentation helper existant
"""
import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain import hub
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.error(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "onea-knowledge-base")

# Cache global
llm_instance = None
embeddings_instance = None
vector_store_instance = None
pinecone_client = None


def get_pinecone_client():
    """Obtenir le client Pinecone"""
    global pinecone_client
    if pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY manquante")
        pinecone_client = Pinecone(api_key=api_key)
    return pinecone_client


def get_gemini_llm():
    """Obtenir l'instance Gemini LLM"""
    global llm_instance
    if llm_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY manquante")
        
        logger.info(f"Initialisation Gemini avec clé: ...{api_key[-4:]}")
        
        # Utiliser gemini-1.5-flash (modèle actuel supporté)
        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
    return llm_instance


def get_gemini_embeddings():
    """Obtenir l'instance des embeddings Gemini"""
    global embeddings_instance
    if embeddings_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY manquante")
        
        embeddings_instance = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    return embeddings_instance


def initialize_pinecone_retriever():
    """Initialiser le retriever Pinecone pour RAG"""
    try:
        pc = get_pinecone_client()
        embeddings = get_gemini_embeddings()
        
        logger.info(f"Connexion à Pinecone index: {INDEX_NAME}")
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        logger.info("Pinecone vectorstore initialisé avec succès")
        return vectorstore
    except Exception as e:
        logger.error(f"Erreur initialisation Pinecone: {e}", exc_info=True)
        return None


def retrieve_sources_from_pinecone(prompt: str, k: int = 5, score_threshold: float = 0.7):
    """Récupérer les sources pertinentes depuis Pinecone"""
    vectorstore = initialize_pinecone_retriever()
    
    if not vectorstore:
        return {"sources": [], "documents": [], "count": 0}
    
    try:
        results = vectorstore.similarity_search_with_score(
            query=prompt,
            k=k
        )
        
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= score_threshold
        ]
        
        if not filtered_results:
            return {"sources": [], "documents": [], "count": 0}
        
        sources = []
        documents = []
        
        for doc, score in filtered_results:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
            documents.append(doc)
        
        # Dédupliquer les sources
        unique_sources = {}
        for source in sources:
            source_url = source["source"]
            if source_url not in unique_sources or source["score"] > unique_sources[source_url]["score"]:
                unique_sources[source_url] = source
        
        return {
            "sources": list(unique_sources.values()),
            "documents": documents,
            "count": len(filtered_results)
        }
    
    except Exception as e:
        logger.error(f"Erreur récupération depuis Pinecone: {e}")
        return {"sources": [], "documents": [], "count": 0}


def create_enhanced_sources_string(sources_data: list) -> str:
    """Créer une chaîne formatée des sources avec scores"""
    if not sources_data:
        return ""
    
    sources_string = "\n\n📚 **Sources utilisées:**\n"
    for i, source in enumerate(sources_data):
        relevance = "🔥" if source["score"] > 0.8 else "⭐" if source["score"] > 0.7 else "📄"
        sources_string += f"{i + 1}. {relevance} {source['source']} (Pertinence: {source['score']:.2f})\n"
    return sources_string


def run_rag_chatbot(query: str, chat_history: List[Dict[str, Any]] = None, context: Dict = None):
    """
    Fonction principale du chatbot RAG
    
    Args:
        query: Question de l'utilisateur
        chat_history: Historique de conversation
        context: Contexte additionnel (station_id, etc.)
    """
    if chat_history is None:
        chat_history = []
    
    try:
        vectorstore = initialize_pinecone_retriever()
        if not vectorstore:
            return run_fallback_response(query, chat_history)
        
        chat = get_gemini_llm()
        
        # Enrichir la requête avec le contexte
        enriched_query = query
        if context and context.get("station_id"):
            enriched_query = f"Station {context['station_id']}: {query}"
        
        # Récupérer les sources pertinentes
        retrieval_results = retrieve_sources_from_pinecone(
            enriched_query,
            k=5,
            score_threshold=0.6
        )
        
        # Prompts depuis LangChain Hub
        try:
            rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        except Exception as e:
            logger.error(f"Erreur chargement prompts: {e}")
            return run_fallback_response(query, chat_history)
        
        # Créer les chaînes RAG
        stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
        
        history_aware_retriever = create_history_aware_retriever(
            llm=chat,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            prompt=rephrase_prompt
        )
        
        qa = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=stuff_documents_chain
        )
        
        # Générer la réponse
        result = qa.invoke(input={"input": enriched_query, "chat_history": chat_history})
        
        # Formater la réponse avec les sources
        formatted_response = result['answer']
        
        if retrieval_results["sources"]:
            sources_string = create_enhanced_sources_string(retrieval_results['sources'])
            formatted_response += sources_string
        
        return {
            "answer": formatted_response,
            "raw_answer": result['answer'],
            "sources": retrieval_results["sources"],
            "retrieval_count": retrieval_results["count"],
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erreur RAG chatbot: {str(e)}")
        return run_fallback_response(query, chat_history)


def run_fallback_response(query: str, chat_history: List[Dict[str, Any]]):
    """Réponse de secours si RAG échoue - Utilise directement Gemini"""
    try:
        chat = get_gemini_llm()
        
        # Système de prompts spécialisé ONEA
        system_context = """Tu es un assistant intelligent pour les agents de l'ONEA (Office National de l'Eau et de l'Assainissement) au Burkina Faso.
        
Tu aides les agents avec:
- Questions sur le fonctionnement des pompes et stations
- Optimisation de la consommation énergétique  
- Détection et résolution d'anomalies
- Interprétation des métriques (efficacité, coûts, consommation)
- Maintenance préventive et interventions
- Stratégies d'optimisation basées sur les tarifs énergétiques

Réponds de manière claire, précise et pratique."""

        prompt_text = f"""{system_context}

Question: {query}

Réponds de manière professionnelle et utile pour un agent de terrain."""

        response = chat.invoke(prompt_text)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer_text,
            "raw_answer": answer_text,
            "sources": [],
            "retrieval_count": 0,
            "success": True,
            "mode": "fallback",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur fallback: {str(e)}", exc_info=True)
        return {
            "answer": "Désolé, je rencontre des difficultés techniques. Veuillez réessayer.",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Exemples de questions prédéfinies pour les agents
PREDEFINED_QUESTIONS = [
    {
        "id": "q1",
        "category": "Optimisation Énergie",
        "question": "Comment réduire la consommation énergétique aux heures de pointe ?",
        "icon": "⚡"
    },
    {
        "id": "q2",
        "category": "Anomalies",
        "question": "Quels sont les signes d'une pompe défaillante ?",
        "icon": "🔧"
    },
    {
        "id": "q3",
        "category": "Maintenance",
        "question": "Quelle est la fréquence recommandée pour la maintenance préventive ?",
        "icon": "🛠️"
    },
    {
        "id": "q4",
        "category": "Coûts",
        "question": "Comment calculer les économies réalisées avec l'optimisation ?",
        "icon": "💰"
    },
    {
        "id": "q5",
        "category": "Performance",
        "question": "Comment améliorer l'efficacité des pompes ?",
        "icon": "📊"
    },
    {
        "id": "q6",
        "category": "Tarification",
        "question": "Quels sont les meilleurs horaires pour pomper de l'eau ?",
        "icon": "⏰"
    }
]


def get_chatbot_suggestions():
    """Obtenir des suggestions de questions pour les agents"""
    return PREDEFINED_QUESTIONS


if __name__ == "__main__":
    # Test du chatbot
    test_query = "Comment optimiser la consommation d'énergie pendant les heures de pointe ?"
    result = run_rag_chatbot(test_query)
    print(f"\nQuestion: {test_query}")
    print(f"\nRéponse: {result['answer']}")
    print(f"\nSources: {len(result['sources'])}")
