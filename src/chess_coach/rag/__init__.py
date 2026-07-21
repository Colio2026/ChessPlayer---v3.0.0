"""Chess coaching RAG (retrieval-augmented) module.

Provides position-aware retrieval of grandmaster annotations from the
annotated PGN corpus, keyed by ECO code and piece-placement similarity.
"""
from .retriever import RAGRetriever

__all__ = ["RAGRetriever"]
