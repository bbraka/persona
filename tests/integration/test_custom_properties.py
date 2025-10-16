"""
Test custom data properties storage and retrieval.

This test verifies that custom properties passed via the custom data endpoint
are stored as individual Neo4j node properties (not just PKG properties).
"""
import pytest
from persona.core.graph_ops import GraphOps
from persona.services.custom_data_service import CustomDataService
from persona.models.schema import (
    CustomGraphUpdate,
    CustomNodeData,
    CustomRelationshipData,
    NodeModel
)


@pytest.mark.asyncio
async def test_custom_properties_are_stored(mock_neo4j_vector_calls):  # pylint: disable=unused-argument
    """
    Test that custom properties are stored as individual Neo4j properties
    when using the custom data endpoint (not as nested PKG properties).
    """
    async with GraphOps() as graph_ops:
        user_id = "test-custom-props-user"

        # Create user first
        await graph_ops.create_user(user_id)

        custom_service = CustomDataService(graph_ops)

        # Create custom data with user profile properties
        test_update = CustomGraphUpdate(
            nodes=[
                CustomNodeData(
                    name="User:test-custom-props-user",
                    perspective="system",
                    embedding=None,
                    properties={
                        "type": "user_profile",
                        "email": "testuser@example.com",
                        "specialty_knowledge_areas": '["AI", "Machine Learning", "NLP"]',
                        "ai_interaction_style": "Coaching and Motivational",
                        "proactivity_level": "High",
                        "age": "28",
                        "location": "San Francisco",
                        "created_at": "2025-10-16T10:00:00Z",
                        "updated_at": "2025-10-16T10:10:00Z"
                    }
                )
            ],
            relationships=[]  # Skip relationships for this test - focus on properties
        )

        # Add custom data
        response = await custom_service.update_custom_data(user_id, test_update)
        assert response["status"] == "success"

        # Verify the node was created with ALL custom properties
        # by querying Neo4j directly with properties() to get all properties
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN properties(n) as props
        """
        driver = graph_ops.neo4j_manager.driver
        assert driver is not None
        async with driver.session() as session:
            result = await session.run(
                query,
                node_name="User:test-custom-props-user",
                user_id=user_id
            )
            record = await result.single()
            assert record is not None, "Node was not created"

            props = record["props"]

            # Verify all custom properties were stored as individual Neo4j properties
            assert props["name"] == "User:test-custom-props-user"
            assert props["UserId"] == user_id
            assert props["type"] == "user_profile"

            # Verify the node has the correct Neo4j label (not "Unknown")
            # Query for labels separately
            labels_query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            RETURN labels(n) as labels
            """
            labels_result = await driver.session().run(
                labels_query,
                node_name="User:test-custom-props-user",
                user_id=user_id
            )
            labels_record = await labels_result.single()
            assert labels_record is not None
            labels = labels_record["labels"]
            assert "user_profile" in labels, f"Expected 'user_profile' label, got {labels}"
            assert "NodeName" in labels
            assert props["email"] == "testuser@example.com"
            assert props["specialty_knowledge_areas"] == '["AI", "Machine Learning", "NLP"]'
            assert props["ai_interaction_style"] == "Coaching and Motivational"
            assert props["proactivity_level"] == "High"
            assert props["age"] == "28"
            assert props["location"] == "San Francisco"
            assert props["created_at"] == "2025-10-16T10:00:00Z"
            assert props["updated_at"] == "2025-10-16T10:10:00Z"

            # Note: Embedding may or may not be present depending on whether
            # the mock includes embedding generation. The important part is
            # that all custom properties were stored.

        # Cleanup
        await graph_ops.delete_user(user_id)


@pytest.mark.asyncio
async def test_ingest_only_stores_pkg_properties(mock_neo4j_vector_calls):  # pylint: disable=unused-argument
    """
    Test that ingested data only stores PKG properties (discipline, bloom_level, confidence)
    and not arbitrary custom properties. This is the existing behavior for ingestion.
    """
    async with GraphOps() as graph_ops:
        user_id = "test-ingest-pkg-user"

        # Create user first
        await graph_ops.create_user(user_id)

        # Create nodes directly via add_nodes (simulating ingestion)
        # with store_custom_properties=False (the default)
        nodes = [
            NodeModel(
                name="Test Concept",
                type="Concept",
                embedding=None,
                properties={
                    "discipline": "Computer Science",
                    "bloom_level": "Understand",
                    "confidence": 0.95,
                    "custom_field": "This should NOT be stored"  # Should be ignored
                }
            )
        ]

        # Add nodes without custom properties flag (default=False for ingestion)
        await graph_ops.add_nodes(nodes, user_id, store_custom_properties=False)

        # Query Neo4j directly to verify only PKG properties were stored
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN properties(n) as props
        """
        driver = graph_ops.neo4j_manager.driver
        assert driver is not None
        async with driver.session() as session:
            result = await session.run(query, node_name="Test Concept", user_id=user_id)
            record = await result.single()
            assert record is not None, "Node was not created"

            props = record["props"]

            # Verify PKG properties were stored
            assert props["discipline"] == "Computer Science"
            assert props["bloom_level"] == "Understand"
            assert props["confidence"] == 0.95

            # Verify custom_field was NOT stored
            assert "custom_field" not in props

        # Cleanup
        await graph_ops.delete_user(user_id)


@pytest.mark.asyncio
async def test_custom_data_with_special_characters_in_keys(mock_neo4j_vector_calls):  # pylint: disable=unused-argument
    """
    Test that custom properties with special characters in keys are properly sanitized
    (hyphens and spaces converted to underscores for Neo4j compatibility).
    """
    async with GraphOps() as graph_ops:
        user_id = "test-special-chars-user"

        # Create user first
        await graph_ops.create_user(user_id)

        custom_service = CustomDataService(graph_ops)

        # Create custom data with property keys containing special characters
        test_update = CustomGraphUpdate(
            nodes=[
                CustomNodeData(
                    name="test-node",
                    embedding=None,
                    properties={
                        "first-name": "John",  # hyphen should become underscore
                        "last name": "Doe",    # space should become underscore
                        "email_address": "john@example.com"  # underscore should stay
                    }
                )
            ],
            relationships=[]
        )

        response = await custom_service.update_custom_data(user_id, test_update)
        assert response["status"] == "success"

        # Verify properties were stored with sanitized keys
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN properties(n) as props
        """
        driver = graph_ops.neo4j_manager.driver
        assert driver is not None
        async with driver.session() as session:
            result = await session.run(query, node_name="test-node", user_id=user_id)
            record = await result.single()
            assert record is not None

            props = record["props"]

            # Verify sanitized keys
            assert props["first_name"] == "John"  # hyphen -> underscore
            assert props["last_name"] == "Doe"    # space -> underscore
            assert props["email_address"] == "john@example.com"  # unchanged

        # Cleanup
        await graph_ops.delete_user(user_id)
