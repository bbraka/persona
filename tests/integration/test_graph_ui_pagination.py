"""
Integration tests for graph-ui-data endpoint pagination functionality.

Tests cursor-based pagination with composite index on (UserId, name).
"""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_pagination_basic_limit(test_client, isolated_graph_ops):
    """Test that limit parameter returns correct number of nodes"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create 20 test nodes
        test_nodes = [
            {"name": f"Node_{i:02d}", "type": "Test", "properties": {}}
            for i in range(20)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request first 10 nodes
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=10")

        assert response.status_code == 200
        data = response.json()

        # Verify node count
        assert len(data["nodes"]) == 10, f"Expected 10 nodes, got {len(data['nodes'])}"

        # Verify pagination metadata exists
        assert "pagination" in data
        pagination = data["pagination"]

        assert pagination["total_nodes"] == 20
        assert pagination["returned_nodes"] == 10
        assert pagination["has_more"] is True
        assert pagination["next_cursor"] is not None
        assert pagination["limit"] == 10

        # Verify nodes are sorted by name
        node_names = [node["name"] for node in data["nodes"]]
        assert node_names == sorted(node_names), "Nodes should be sorted by name"


async def test_pagination_cursor_sequential(test_client, isolated_graph_ops):
    """Test cursor-based pagination through multiple pages"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create 25 test nodes
        test_nodes = [
            {"name": f"Test_{chr(65 + i)}", "type": "Pagination", "properties": {}}
            for i in range(25)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Page 1: First 10 nodes
        response1 = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=10")
        assert response1.status_code == 200
        data1 = response1.json()

        assert len(data1["nodes"]) == 10
        assert data1["pagination"]["has_more"] is True
        cursor1 = data1["pagination"]["next_cursor"]
        assert cursor1 is not None

        page1_names = [node["name"] for node in data1["nodes"]]

        # Page 2: Next 10 nodes using cursor
        response2 = test_client.get(
            f"/api/v1/users/{user_id}/graph-ui-data?limit=10&cursor={cursor1}"
        )
        assert response2.status_code == 200
        data2 = response2.json()

        assert len(data2["nodes"]) == 10
        assert data2["pagination"]["has_more"] is True
        cursor2 = data2["pagination"]["next_cursor"]

        page2_names = [node["name"] for node in data2["nodes"]]

        # Page 3: Last 5 nodes
        response3 = test_client.get(
            f"/api/v1/users/{user_id}/graph-ui-data?limit=10&cursor={cursor2}"
        )
        assert response3.status_code == 200
        data3 = response3.json()

        assert len(data3["nodes"]) == 5  # Only 5 nodes left
        assert data3["pagination"]["has_more"] is False  # No more pages
        assert data3["pagination"]["next_cursor"] is None  # No next cursor

        page3_names = [node["name"] for node in data3["nodes"]]

        # Verify no overlapping nodes between pages
        assert len(set(page1_names) & set(page2_names)) == 0, "Page 1 and 2 should not overlap"
        assert len(set(page2_names) & set(page3_names)) == 0, "Page 2 and 3 should not overlap"
        assert len(set(page1_names) & set(page3_names)) == 0, "Page 1 and 3 should not overlap"

        # Verify all nodes are accounted for
        all_names = page1_names + page2_names + page3_names
        assert len(all_names) == 25
        assert len(set(all_names)) == 25, "All nodes should be unique"


async def test_pagination_backward_compatibility(test_client, isolated_graph_ops):
    """Test that omitting limit returns all nodes (backward compatible)"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create 15 test nodes
        test_nodes = [
            {"name": f"BackCompat_{i}", "type": "Test", "properties": {}}
            for i in range(15)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request without limit (should return all nodes)
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

        assert response.status_code == 200
        data = response.json()

        # Should return all 15 nodes
        assert len(data["nodes"]) == 15

        # Pagination metadata should still be present but indicate all data returned
        # Note: The implementation includes pagination metadata even without limit
        # If you want to exclude it when no limit, update the service


async def test_pagination_with_filters(test_client, isolated_graph_ops):
    """Test pagination works correctly with entity ID filters"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create nodes with different book_ids
        test_nodes = [
            {"name": f"Book1_Node_{i}", "type": "Test", "properties": {}, "book_id": [1]}
            for i in range(10)
        ] + [
            {"name": f"Book2_Node_{i}", "type": "Test", "properties": {}, "book_id": [2]}
            for i in range(10)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request only book_id=1 with limit
        response = test_client.get(
            f"/api/v1/users/{user_id}/graph-ui-data?book_id=1&limit=5"
        )

        assert response.status_code == 200
        data = response.json()

        # Should return 5 nodes from book 1
        assert len(data["nodes"]) == 5
        assert data["pagination"]["total_nodes"] == 10  # Total for book_id=1
        assert data["pagination"]["has_more"] is True

        # Verify all returned nodes belong to book_id=1
        for node in data["nodes"]:
            assert 1 in node.get("book_id", []), f"Node {node['name']} should belong to book_id=1"


async def test_pagination_cursor_ordering(test_client, isolated_graph_ops):
    """Test that cursor properly respects alphabetical ordering"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create nodes with specific names to test ordering
        test_nodes = [
            {"name": "Alpha", "type": "Test", "properties": {}},
            {"name": "Beta", "type": "Test", "properties": {}},
            {"name": "Gamma", "type": "Test", "properties": {}},
            {"name": "Delta", "type": "Test", "properties": {}},
            {"name": "Epsilon", "type": "Test", "properties": {}},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Get first 2 nodes
        response1 = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=2")
        data1 = response1.json()

        assert data1["nodes"][0]["name"] == "Alpha"
        assert data1["nodes"][1]["name"] == "Beta"
        assert data1["pagination"]["next_cursor"] == "Beta"

        # Get next page after "Beta"
        response2 = test_client.get(
            f"/api/v1/users/{user_id}/graph-ui-data?limit=2&cursor=Beta"
        )
        data2 = response2.json()

        # Should start after "Beta" (not including Beta)
        assert data2["nodes"][0]["name"] == "Delta"
        assert data2["nodes"][1]["name"] == "Epsilon"


async def test_pagination_empty_result(test_client, isolated_graph_ops):
    """Test pagination with no matching nodes"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Don't create any nodes

        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=10")

        assert response.status_code == 200
        data = response.json()

        assert len(data["nodes"]) == 0
        assert data["pagination"]["total_nodes"] == 0
        assert data["pagination"]["returned_nodes"] == 0
        assert data["pagination"]["has_more"] is False
        assert data["pagination"]["next_cursor"] is None


async def test_pagination_exact_page_size(test_client, isolated_graph_ops):
    """Test pagination when total nodes exactly match limit"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create exactly 10 nodes
        test_nodes = [
            {"name": f"Exact_{i:02d}", "type": "Test", "properties": {}}
            for i in range(10)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request with limit=10 (exact match)
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=10")

        assert response.status_code == 200
        data = response.json()

        assert len(data["nodes"]) == 10
        assert data["pagination"]["total_nodes"] == 10
        assert data["pagination"]["returned_nodes"] == 10
        assert data["pagination"]["has_more"] is False  # No more pages
        assert data["pagination"]["next_cursor"] is None


async def test_pagination_relationships_filtered(test_client, isolated_graph_ops):
    """Test that relationships are only returned for nodes in current page"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create nodes and relationships
        test_nodes = [
            {"name": "Node_A", "type": "Test", "properties": {}},
            {"name": "Node_B", "type": "Test", "properties": {}},
            {"name": "Node_C", "type": "Test", "properties": {}},
            {"name": "Node_D", "type": "Test", "properties": {}},
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Create relationships: A->B, B->C, C->D
        test_relationships = [
            {"source": "Node_A", "target": "Node_B", "relation": "CONNECTS_TO"},
            {"source": "Node_B", "target": "Node_C", "relation": "CONNECTS_TO"},
            {"source": "Node_C", "target": "Node_D", "relation": "CONNECTS_TO"},
        ]
        await graph_ops.neo4j_manager.create_relationships(test_relationships, user_id)

        # Get only first 2 nodes (Node_A, Node_B)
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=2")

        assert response.status_code == 200
        data = response.json()

        # Should return 2 nodes
        assert len(data["nodes"]) == 2
        node_names = [node["name"] for node in data["nodes"]]
        assert "Node_A" in node_names
        assert "Node_B" in node_names

        # Should only return relationship A->B (both nodes in page)
        # Should NOT include B->C (C not in page) or C->D (neither in page)
        assert len(data["relationships"]) == 1
        rel = data["relationships"][0]
        assert rel["source"] == "Node_A"
        assert rel["target"] == "Node_B"


async def test_pagination_large_limit(test_client, isolated_graph_ops):
    """Test pagination with limit larger than total nodes"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create 5 nodes
        test_nodes = [
            {"name": f"Small_{i}", "type": "Test", "properties": {}}
            for i in range(5)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request with limit=100 (much larger than 5)
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?limit=100")

        assert response.status_code == 200
        data = response.json()

        # Should return all 5 nodes
        assert len(data["nodes"]) == 5
        assert data["pagination"]["total_nodes"] == 5
        assert data["pagination"]["returned_nodes"] == 5
        assert data["pagination"]["has_more"] is False
        assert data["pagination"]["next_cursor"] is None


async def test_pagination_with_date_filter(test_client, isolated_graph_ops):
    """Test pagination works with date filtering"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Create nodes with different created_at dates
        from datetime import datetime
        test_nodes = [
            {
                "name": f"Feb24_Node_{i}",
                "type": "Test",
                "properties": {},
                "created_at": "2025-02-24T10:00:00+00:00"
            }
            for i in range(5)
        ] + [
            {
                "name": f"Feb25_Node_{i}",
                "type": "Test",
                "properties": {},
                "created_at": "2025-02-25T10:00:00+00:00"
            }
            for i in range(5)
        ]
        await graph_ops.neo4j_manager.create_nodes(test_nodes, user_id)

        # Request only Feb 24 nodes with pagination
        response = test_client.get(
            f"/api/v1/users/{user_id}/graph-ui-data?date_from=2025-02-24&date_to=2025-02-24&limit=3"
        )

        assert response.status_code == 200
        data = response.json()

        # Should return 3 nodes from Feb 24
        assert len(data["nodes"]) == 3
        assert data["pagination"]["total_nodes"] == 5  # Total for Feb 24
        assert data["pagination"]["has_more"] is True

        # Verify all returned nodes are from Feb 24
        for node in data["nodes"]:
            assert node["created_at"].startswith("2025-02-24")
