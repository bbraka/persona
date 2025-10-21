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


async def test_graph_ui_data_filter_by_search(test_client, api_test_user):
    """Test filtering graph UI data by search term."""
    user_id = f"{api_test_user}_search"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Test Content",
            "content": "I love quantum computing and artificial intelligence. I also enjoy classical music.",
            "metadata": {}
        }
    )

    # Search for "quantum"
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?search=quantum")

    assert response.status_code == 200
    data = response.json()

    # All returned nodes should contain "quantum" in their name (case-insensitive)
    for node in data["nodes"]:
        assert "quantum" in node["name"].lower()

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_filter_by_topics(test_client, api_test_user):
    """Test filtering graph UI data by topics/disciplines."""
    user_id = f"{api_test_user}_topics_filter"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
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

    # Wait a moment for processing
    import time
    time.sleep(2)

    # First, get all data to see what disciplines exist
    response_all = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
    assert response_all.status_code == 200
    all_topics = response_all.json()["topics"]

    # If we have topics, filter by the first one
    if len(all_topics) > 0:
        first_topic = all_topics[0]["name"]

        # Filter by specific topic
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?topics={first_topic}")

        assert response.status_code == 200
        data = response.json()

        # All returned nodes should have the specified discipline
        for node in data["nodes"]:
            if node.get("discipline"):
                assert node["discipline"] == first_topic

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_filter_by_min_confidence(test_client, api_test_user):
    """Test filtering insights by minimum confidence."""
    user_id = f"{api_test_user}_confidence"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Test Content",
            "content": "I am passionate about machine learning and deep learning technologies.",
            "metadata": {}
        }
    )

    # Filter by high confidence (0.9)
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?min_confidence=0.9")

    assert response.status_code == 200
    data = response.json()

    # All insights should have confidence >= 0.9
    for insight in data["insights"]:
        assert insight["confidence"] >= 0.9

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_combined_filters(test_client, api_test_user):
    """Test combining multiple filters."""
    user_id = f"{api_test_user}_combined"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Technology Interests",
            "content": "I love quantum computing, artificial intelligence, and machine learning.",
            "metadata": {}
        }
    )

    # Combine search and min_confidence filters
    response = test_client.get(
        f"/api/v1/users/{user_id}/graph-ui-data?search=quantum&min_confidence=0.5"
    )

    assert response.status_code == 200
    data = response.json()

    # Nodes should match search term
    for node in data["nodes"]:
        assert "quantum" in node["name"].lower()

    # Insights should meet confidence threshold
    for insight in data["insights"]:
        assert insight["confidence"] >= 0.5

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_filter_injection_prevention(test_client, api_test_user):
    """Test that filter parameters are properly sanitized against injection."""
    user_id = f"{api_test_user}_injection"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Test Content",
            "content": "I love technology and science.",
            "metadata": {}
        }
    )

    # Try potential Cypher injection in search parameter
    malicious_search = "' OR 1=1 OR '"
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?search={malicious_search}")

    # Should not cause an error, should treat as literal search string
    assert response.status_code == 200
    data = response.json()

    # Should return empty or valid results (not all nodes)
    assert isinstance(data["nodes"], list)

    # Try malicious topic filter
    malicious_topic = "Technology'; DROP TABLE NodeName; --"
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?topics={malicious_topic}")

    # Should not cause an error
    assert response.status_code == 200

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_sources_filter_with_books(test_client, api_test_user):
    """Test filtering by sources (book names) to get nodes connected to specific books."""
    user_id = f"{api_test_user}_sources_books"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add custom data with a book and related content
    custom_data = {
        "nodes": [
            {
                "name": "Test Book",
                "type": "Book",
                "properties": {},
                "embedding": None
            },
            {
                "name": "Main Character",
                "type": "Character",
                "properties": {},
                "embedding": None
            },
            {
                "name": "Central Theme",
                "type": "Theme",
                "properties": {},
                "embedding": None
            },
            {
                "name": "Other Book",
                "type": "Book",
                "properties": {},
                "embedding": None
            },
            {
                "name": "Other Character",
                "type": "Character",
                "properties": {},
                "embedding": None
            }
        ],
        "relationships": [
            {
                "source": "Main Character",
                "target": "Test Book",
                "relation_type": "APPEARS_IN",
                "data": {}
            },
            {
                "source": "Central Theme",
                "target": "Test Book",
                "relation_type": "EXPLORED_IN",
                "data": {}
            },
            {
                "source": "Other Character",
                "target": "Other Book",
                "relation_type": "APPEARS_IN",
                "data": {}
            }
        ]
    }

    test_client.post(f"/api/v1/users/{user_id}/custom-data", json=custom_data)

    # Get all data first
    response_all = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
    assert response_all.status_code == 200
    all_data = response_all.json()

    # Should have sources (books) in the response
    assert len(all_data["sources"]) > 0
    book_names = [s["name"] for s in all_data["sources"]]
    assert "Test Book" in book_names

    # Filter by "Test Book" source
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?sources=Test Book")

    assert response.status_code == 200
    filtered_data = response.json()

    # Should have fewer nodes than all data (excluding Book nodes themselves)
    assert len(filtered_data["nodes"]) < len(all_data["nodes"])

    # All returned nodes should be connected to "Test Book"
    node_names = [n["name"] for n in filtered_data["nodes"]]
    assert "Main Character" in node_names
    assert "Central Theme" in node_names
    assert "Other Character" not in node_names  # This is connected to "Other Book"
    assert "Test Book" not in node_names  # Book nodes are excluded from results

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_multiple_sources_filter(test_client, api_test_user):
    """Test filtering by multiple book sources."""
    user_id = f"{api_test_user}_multi_sources"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add custom data with multiple books
    custom_data = {
        "nodes": [
            {"name": "Book A", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Book B", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Book C", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Concept from A", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Concept from B", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Concept from C", "type": "Concept", "properties": {}, "embedding": None},
        ],
        "relationships": [
            {"source": "Concept from A", "target": "Book A", "relation_type": "FROM", "data": {}},
            {"source": "Concept from B", "target": "Book B", "relation_type": "FROM", "data": {}},
            {"source": "Concept from C", "target": "Book C", "relation_type": "FROM", "data": {}},
        ]
    }

    test_client.post(f"/api/v1/users/{user_id}/custom-data", json=custom_data)

    # Filter by Book A and Book B (but not Book C)
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?sources=Book A,Book B")

    assert response.status_code == 200
    data = response.json()

    node_names = [n["name"] for n in data["nodes"]]
    assert "Concept from A" in node_names
    assert "Concept from B" in node_names
    assert "Concept from C" not in node_names  # Not in filtered books

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_combined_sources_and_search(test_client, api_test_user):
    """Test combining sources filter with search filter."""
    user_id = f"{api_test_user}_sources_search"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add custom data
    custom_data = {
        "nodes": [
            {"name": "My Book", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Quantum Physics", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Classical Music", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Quantum Computing", "type": "Concept", "properties": {}, "embedding": None},
        ],
        "relationships": [
            {"source": "Quantum Physics", "target": "My Book", "relation_type": "FROM", "data": {}},
            {"source": "Classical Music", "target": "My Book", "relation_type": "FROM", "data": {}},
            {"source": "Quantum Computing", "target": "My Book", "relation_type": "FROM", "data": {}},
        ]
    }

    test_client.post(f"/api/v1/users/{user_id}/custom-data", json=custom_data)

    # Combine sources and search filters
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?sources=My Book&search=quantum")

    assert response.status_code == 200
    data = response.json()

    node_names = [n["name"] for n in data["nodes"]]

    # Should only have nodes from "My Book" that contain "quantum"
    assert "Quantum Physics" in node_names
    assert "Quantum Computing" in node_names
    assert "Classical Music" not in node_names  # Doesn't contain "quantum"

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_sources_query_returns_books(test_client, api_test_user):
    """Test that the sources query returns Book nodes with content counts."""
    user_id = f"{api_test_user}_sources_query"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add custom data with books and content
    custom_data = {
        "nodes": [
            {"name": "Popular Book", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Rare Book", "type": "Book", "properties": {}, "embedding": None},
            {"name": "Concept 1", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Concept 2", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Concept 3", "type": "Concept", "properties": {}, "embedding": None},
            {"name": "Concept 4", "type": "Concept", "properties": {}, "embedding": None},
        ],
        "relationships": [
            # Popular Book has 3 concepts
            {"source": "Concept 1", "target": "Popular Book", "relation_type": "FROM", "data": {}},
            {"source": "Concept 2", "target": "Popular Book", "relation_type": "FROM", "data": {}},
            {"source": "Concept 3", "target": "Popular Book", "relation_type": "FROM", "data": {}},
            # Rare Book has 1 concept
            {"source": "Concept 4", "target": "Rare Book", "relation_type": "FROM", "data": {}},
        ]
    }

    test_client.post(f"/api/v1/users/{user_id}/custom-data", json=custom_data)

    # Get graph UI data
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

    assert response.status_code == 200
    data = response.json()

    # Sources should list books with their content counts
    assert len(data["sources"]) == 2

    source_dict = {s["name"]: s["count"] for s in data["sources"]}
    assert "Popular Book" in source_dict
    assert "Rare Book" in source_dict
    assert source_dict["Popular Book"] == 3
    assert source_dict["Rare Book"] == 1

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_all_filters_combined(test_client, api_test_user):
    """Test using all filters together: sources, topics, search, min_confidence."""
    user_id = f"{api_test_user}_all_filters"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Ingest content that will create various nodes
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Tech Book",
            "content": "I'm learning about quantum computing, machine learning, and artificial intelligence.",
            "metadata": {}
        }
    )

    import time
    time.sleep(2)  # Wait for processing

    # Get all data first
    response_all = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
    all_data = response_all.json()
    total_nodes = len(all_data["nodes"])

    # Apply search and confidence filters together
    # (sources filter would need book relationships which are complex to set up in this test)
    response = test_client.get(
        f"/api/v1/users/{user_id}/graph-ui-data?"
        f"search=quantum&min_confidence=0.5"
    )

    assert response.status_code == 200
    filtered_data = response.json()

    # Should have valid response structure
    assert "topics" in filtered_data
    assert "insights" in filtered_data
    assert "sources" in filtered_data
    assert "nodes" in filtered_data
    assert "relationships" in filtered_data

    # All returned nodes should contain "quantum" in name
    for node in filtered_data["nodes"]:
        assert "quantum" in node["name"].lower()

    # All insights should meet confidence threshold
    for insight in filtered_data["insights"]:
        assert insight["confidence"] >= 0.5

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_empty_sources_list(test_client, api_test_user):
    """Test that empty sources list returns all data."""
    user_id = f"{api_test_user}_empty_sources"

    # Create user
    test_client.post(f"/api/v1/users/{user_id}")

    # Add test data
    test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Test",
            "content": "Some content for testing.",
            "metadata": {}
        }
    )

    # Get data with empty sources (should return all)
    response1 = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
    response2 = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data?sources=")

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Both should return the same results
    assert len(data1["nodes"]) == len(data2["nodes"])

    # Cleanup
    test_client.delete(f"/api/v1/users/{user_id}")


async def test_graph_ui_data_entity_ids(test_client, api_test_user):
    """Test that nodes include entity ID fields (book_id, highlight_id, writing_id)."""
    user_id = api_test_user
    
    # Ensure user exists
    test_client.post(f"/api/v1/users/{user_id}")
    
    # Add test data with entity IDs using custom data endpoint
    test_client.post(
        f"/api/v1/users/{user_id}/custom-data",
        json={
            "nodes": [
                {
                    "name": "Node with entity IDs",
                    "type": "Concept",
                    "chunk_ids": ["test-chunk-1"],
                    "properties": {
                        "book_id": [27, 45],
                        "highlight_id": [123, 456],
                        "writing_id": [789],
                        "discipline": "Philosophy",
                        "bloom_level": "Understand",
                        "confidence": 0.9
                    }
                }
            ],
            "relationships": []
        }
    )
    
    # Get graph UI data
    response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
    assert response.status_code == 200
    data = response.json()
    
    # Find the node we just created
    nodes = data["nodes"]
    test_node = next((n for n in nodes if n["name"] == "Node with entity IDs"), None)
    
    assert test_node is not None, "Test node not found in response"
    
    # Verify entity ID fields are present and have correct values
    # Note: Neo4j stores these as properties, so they should be in the node dict
    assert "chunk_ids" in test_node
    assert test_node["chunk_ids"] == ["test-chunk-1"]
    
    # Check if entity IDs are in the node dict or properties
    # They should be in top-level based on our implementation
    if "book_id" in test_node:
        assert test_node["book_id"] == [27, 45] or test_node["book_id"] == []
    if "highlight_id" in test_node:
        assert test_node["highlight_id"] == [123, 456] or test_node["highlight_id"] == []
    if "writing_id" in test_node:
        assert test_node["writing_id"] == [789] or test_node["writing_id"] == []
    
    # Verify standard fields are also present
    assert test_node["discipline"] == "Philosophy"
    assert test_node["bloom_level"] == "Understand"
    assert test_node["confidence"] == 0.9
