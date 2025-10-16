"""
Integration tests for the graph UI data endpoint.
"""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_graph_ui_data_structure(test_client, api_test_user):
    """Test that the graph UI data endpoint returns the correct structure."""
    user_id = api_test_user

    # Ensure user exists
    test_client.post(f"/api/v1/users/{user_id}")

    # Add some test data via ingest
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Test Content",
            "content": "I love machine learning and artificial intelligence. I'm currently studying deep learning.",
            "metadata": {}
        }
    )

    # Get graph UI data
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "topics" in data
    assert "insights" in data
    assert "sources" in data
    assert "nodes" in data
    assert "relationships" in data

    # Verify topics structure
    assert isinstance(data["topics"], list)
    if len(data["topics"]) > 0:
        topic = data["topics"][0]
        assert "name" in topic
        assert "entity_count" in topic
        assert "relationship_count" in topic
        assert "bloom_distribution" in topic

    # Verify insights structure
    assert isinstance(data["insights"], list)
    if len(data["insights"]) > 0:
        insight = data["insights"][0]
        assert "description" in insight
        assert "confidence" in insight
        assert "source" in insight
        # Confidence should be >= 0.7 for insights
        assert insight["confidence"] >= 0.7

    # Verify sources structure
    assert isinstance(data["sources"], list)
    if len(data["sources"]) > 0:
        source = data["sources"][0]
        assert "name" in source
        assert "count" in source

    # Verify nodes structure
    assert isinstance(data["nodes"], list)
    if len(data["nodes"]) > 0:
        node = data["nodes"][0]
        assert "id" in node
        assert "name" in node
        assert "type" in node or node.get("type") is None
        assert "properties" in node

    # Verify relationships structure
    assert isinstance(data["relationships"], list)
    if len(data["relationships"]) > 0:
        rel = data["relationships"][0]
        assert "source" in rel
        assert "target" in rel
        assert "relation" in rel


async def test_graph_ui_data_empty_graph(test_client, api_test_user):
    """Test graph UI data endpoint with a user that has no data."""
    user_id = f"{api_test_user}_empty"

    # Create user but don't add any data
    test_client.post(f"/api/v1/users/{user_id}")

    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

    assert response.status_code == 200
    data = response.json()

    # All arrays should be empty
    assert data["topics"] == []
    assert data["insights"] == []
    assert data["sources"] == []
    assert data["nodes"] == []
    assert data["relationships"] == []

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_nonexistent_user(test_client):
    """Test that requesting graph UI data for a non-existent user returns 404."""
    response = test_client.get("/api/v1/users/nonexistent_user_12345/graph-ui-data")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


async def test_graph_ui_data_with_custom_data(test_client, api_test_user):
    """Test graph UI data with custom data that has additional properties."""
    user_id = f"{api_test_user}_custom"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add custom data with extra properties
    custom_data = {
        "nodes": [
            {
                "name": "User",
                "perspective": "user_profile",
                "properties": {
                    "type": "user_profile",
                    "email": "test@example.com",
                    "ai_interaction_style": "detailed",
                    "preferred_language": "English"
                },
                "embedding": None
            }
        ],
        "relationships": []
    }

    test_client.post(
        f"/api/v1/users/{user_id}/custom-data",
        json=custom_data
    )

    # Get graph UI data
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

    assert response.status_code == 200
    data = response.json()

    # Should have at least one node
    assert len(data["nodes"]) > 0

    # Find our custom node
    user_node = next((n for n in data["nodes"] if n["name"] == "User"), None)
    assert user_node is not None

    # Verify custom properties are included
    props = user_node["properties"]
    assert "email" in props
    assert props["email"] == "test@example.com"
    assert "ai_interaction_style" in props
    assert props["ai_interaction_style"] == "detailed"

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_topics_aggregation(test_client, api_test_user):
    """Test that topics are properly aggregated by discipline."""
    user_id = f"{api_test_user}_topics"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Ingest content that should create multiple topics
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Learning Journey",
            "content": """
            I'm passionate about computer science and programming.
            I also enjoy studying psychology and human behavior.
            Recently I've been learning about physics and quantum mechanics.
            """,
            "metadata": {}
        }
    )

    # Get graph UI data
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

    assert response.status_code == 200
    data = response.json()

    # Should have topics from different disciplines
    topics = data["topics"]
    assert len(topics) > 0

    # Each topic should have the required fields
    for topic in topics:
        assert topic["entity_count"] > 0
        assert isinstance(topic["bloom_distribution"], dict)
        assert topic["relationship_count"] >= 0

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")
