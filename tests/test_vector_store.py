"""Tests for source-specific vector-store operations."""

from unittest.mock import Mock

from src.retrieval.vector_store import VectorStore


def test_delete_by_source_removes_only_matching_chunk_ids():
    collection = Mock()
    collection.get.return_value = {"ids": ["policy_chunk_1", "policy_chunk_2"]}
    vector_store = VectorStore.__new__(VectorStore)
    vector_store.collection = collection

    deleted_count = vector_store.delete_by_source("policy.pdf")

    assert deleted_count == 2
    collection.get.assert_called_once_with(
        where={"source": "policy.pdf"},
        include=["metadatas"]
    )
    collection.delete.assert_called_once_with(
        ids=["policy_chunk_1", "policy_chunk_2"]
    )


def test_delete_by_source_skips_delete_when_source_has_no_chunks():
    collection = Mock()
    collection.get.return_value = {"ids": []}
    vector_store = VectorStore.__new__(VectorStore)
    vector_store.collection = collection

    assert vector_store.delete_by_source("missing.pdf") == 0
    collection.delete.assert_not_called()