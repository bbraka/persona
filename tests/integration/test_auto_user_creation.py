"""
Integration tests for auto-user creation functionality.

Tests that all API endpoints automatically create users when they don't exist,
ensuring users never get "user doesn't exist" errors.
"""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


async def test_ingest_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /ingest endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make ingest request without creating user first
        response = test_client.post(
            f"/api/v1/users/{user_id}/ingest",
            json={
                "title": "Auto-create test",
                "content": "Testing automatic user creation on ingest.",
                "metadata": {}
            }
        )

        # Should succeed (user auto-created)
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.json()}"
        assert "ingested successfully" in response.json()["message"]

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_rag_query_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /rag/query endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make RAG query without creating user first
        response = test_client.post(
            f"/api/v1/users/{user_id}/rag/query",
            json={"query": "What is this about?"}
        )

        # Should succeed (user auto-created)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        assert "answer" in response.json()

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_rag_query_vector_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /rag/query-vector endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make vector query without creating user first
        response = test_client.post(
            f"/api/v1/users/{user_id}/rag/query-vector",
            json={"query": "Test vector query"}
        )

        # Should succeed (user auto-created)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        assert "query" in response.json()
        assert "response" in response.json()

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_ask_insights_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /ask endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make ask request without creating user first
        response = test_client.post(
            f"/api/v1/users/{user_id}/ask",
            json={
                "query": "What are the main topics?",
                "output_schema": {
                    "topics": ["topic1"],
                    "summary": "summary"
                }
            }
        )

        # Should succeed (user auto-created)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        assert response.json() is not None

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_custom_data_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /custom-data endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make custom data request without creating user first
        response = test_client.post(
            f"/api/v1/users/{user_id}/custom-data",
            json={
                "nodes": [
                    {
                        "name": "Auto-created Test Node",
                        "properties": {"test": "value"},
                        "perspective": "test"
                    }
                ],
                "relationships": []
            }
        )

        # Should succeed (user auto-created)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        assert "status" in response.json()

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_graph_ui_data_auto_creates_user(test_client, isolated_graph_ops):
    """Test that /graph-ui-data endpoint auto-creates user if they don't exist"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Make graph UI data request without creating user first
        response = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")

        # Should succeed (user auto-created)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        result = response.json()
        assert "nodes" in result
        assert "relationships" in result

        # Verify user was created
        assert await graph_ops.user_exists(user_id), "User should have been auto-created"


async def test_multiple_endpoints_same_user(test_client, isolated_graph_ops):
    """Test that multiple endpoints can be called for same user without errors"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # First call - should auto-create user
        response1 = test_client.post(
            f"/api/v1/users/{user_id}/ingest",
            json={
                "title": "First call",
                "content": "Testing first endpoint call.",
                "metadata": {}
            }
        )
        assert response1.status_code == 201

        # User should now exist
        assert await graph_ops.user_exists(user_id), "User should exist after first call"

        # Second call to different endpoint - should not error
        response2 = test_client.post(
            f"/api/v1/users/{user_id}/rag/query",
            json={"query": "What did I just ingest?"}
        )
        assert response2.status_code == 200

        # Third call to another endpoint - should not error
        response3 = test_client.get(f"/api/v1/users/{user_id}/graph-ui-data")
        assert response3.status_code == 200


async def test_explicit_user_creation_still_works(test_client, isolated_graph_ops):
    """Test that explicit POST /users/{user_id} still works as expected"""
    async for graph_ops, user_id in isolated_graph_ops:
        # Delete the user to ensure they don't exist
        await graph_ops.delete_user(user_id)

        # Verify user doesn't exist
        assert not await graph_ops.user_exists(user_id), "User should not exist before test"

        # Explicitly create user
        response = test_client.post(f"/api/v1/users/{user_id}")
        assert response.status_code == 201
        assert "created successfully" in response.json()["message"]

        # Verify user exists
        assert await graph_ops.user_exists(user_id), "User should exist after creation"

        # Try to create again - should return 200 with "already exists"
        response2 = test_client.post(f"/api/v1/users/{user_id}")
        assert response2.status_code == 200
        assert "already exists" in response2.json()["message"]


async def test_delete_user_still_requires_existence(test_client):
    """Test that DELETE /users/{user_id} still returns 404 for non-existent users"""
    non_existent_user = "definitely-does-not-exist-12345"

    # Try to delete non-existent user
    response = test_client.delete(f"/api/v1/users/{non_existent_user}")

    # Should return 404
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
