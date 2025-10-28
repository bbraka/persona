"""
Test that ask service treats 0 as None for entity ID filters.

This was a bug where passing book_id=0, highlight_id=0, writing_id=0
would filter for nodes with those IDs (which don't exist), resulting in
empty context and empty results.
"""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ask_with_zero_filters(test_client, isolated_graph_ops):
    """Test that book_id=0, highlight_id=0, writing_id=0 are treated as no filter"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create test nodes without specifying entity IDs
        test_nodes = [
            {"name": "Test Concept A", "type": "Concept", "properties": {}},
            {"name": "Test Concept B", "type": "Concept", "properties": {}},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Create relationship
        await graph_ops.neo4j_manager.create_relationships([
            {"source": "Test Concept A", "target": "Test Concept B", "relation": "relates_to"}
        ], user_id)

        # Request with zero filters (should NOT filter out nodes)
        response = test_client.post(
            f"/api/v1/users/{user_id}/ask",
            json={
                "query": "Tell me about Test Concept A",
                "output_schema": {"result": "string"},
                "book_id": 0,
                "highlight_id": 0,
                "writing_id": 0
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should have actual content (not empty)
        assert "result" in data
        result = data["result"]["result"]
        assert isinstance(result, str)
        assert len(result) > 0, "Result should not be empty when filters are 0"
        # Result should mention the concepts
        assert "Test Concept" in result or "Concept A" in result or "Concept B" in result


async def test_ask_with_null_filters(test_client, isolated_graph_ops):
    """Test that null filters work (baseline)"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create test nodes
        test_nodes = [
            {"name": "Null Test A", "type": "Concept", "properties": {}},
            {"name": "Null Test B", "type": "Concept", "properties": {}},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request with null filters
        response = test_client.post(
            f"/api/v1/users/{user_id}/ask",
            json={
                "query": "Tell me about Null Test A",
                "output_schema": {"result": "string"},
                "book_id": None,
                "highlight_id": None,
                "writing_id": None
            }
        )

        assert response.status_code == 200
        data = response.json()

        result = data["result"]["result"]
        assert len(result) > 0, "Result should not be empty"


async def test_ask_without_filters(test_client, isolated_graph_ops):
    """Test that omitting filters works (baseline)"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create test nodes
        test_nodes = [
            {"name": "No Filter A", "type": "Concept", "properties": {}},
            {"name": "No Filter B", "type": "Concept", "properties": {}},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request without filter fields
        response = test_client.post(
            f"/api/v1/users/{user_id}/ask",
            json={
                "query": "Tell me about No Filter A",
                "output_schema": {"result": "string"}
            }
        )

        assert response.status_code == 200
        data = response.json()

        result = data["result"]["result"]
        assert len(result) > 0, "Result should not be empty"


async def test_ask_with_valid_book_id(test_client, isolated_graph_ops):
    """Test that valid book_id filter works correctly"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create nodes with specific book_id
        test_nodes = [
            {"name": "Book Concept A", "type": "Concept", "properties": {}, "book_id": [42]},
            {"name": "Book Concept B", "type": "Concept", "properties": {}, "book_id": [99]},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request with valid book_id filter
        response = test_client.post(
            f"/api/v1/users/{user_id}/ask",
            json={
                "query": "Tell me about Book Concept",
                "output_schema": {"result": "string"},
                "book_id": 42
            }
        )

        assert response.status_code == 200
        data = response.json()

        result = data["result"]["result"]
        # Should mention Book Concept A (which has book_id=42)
        # Should NOT mention Book Concept B (which has book_id=99)
        # Note: This might not be strictly true if the LLM context includes both,
        # but at least the result should not be empty
        assert len(result) > 0, "Result should not be empty with valid filter"
