"""
Test chunk_id extraction and storage.

This test verifies that chunk_id is properly extracted from text content by the LLM
and stored in Neo4j nodes when ingesting book content.
"""
import pytest
from persona.core.constructor import GraphConstructor
from persona.core.graph_ops import GraphOps
from persona.models.schema import UnstructuredData
from unittest.mock import patch, AsyncMock
from persona.llm.llm_graph import Node as LLMNode


@pytest.fixture(autouse=True)
def mock_llm_with_chunk_id(monkeypatch, request):
    """Override the autouse fixture to return nodes with chunk_id"""
    # Skip this fixture if the test doesn't need it
    if 'no_mock_llm' in request.keywords:
        return

    # Only apply to tests in this file
    if 'test_chunk_id_extracted_from_llm_and_stored' not in request.node.name:
        return

    mock_llm_nodes = [
        LLMNode(
            name="Vector Databases",
            type="Concept",
            chunk_id="book_chunk_123",
            discipline="Computer Science",
            bloom_level="Understand",
            confidence=0.9
        ),
        LLMNode(
            name="Similarity Search",
            type="Concept",
            chunk_id="book_chunk_123",
            discipline="Computer Science",
            bloom_level="Apply",
            confidence=0.85
        )
    ]

    async def mock_get_nodes(*args, **kwargs):
        return mock_llm_nodes

    async def mock_get_relationships(*args, **kwargs):
        return ([], {})

    # This will override the autouse fixture from conftest
    monkeypatch.setattr("persona.llm.llm_graph.get_nodes", mock_get_nodes)
    monkeypatch.setattr("persona.llm.llm_graph.get_relationships", mock_get_relationships)
    return mock_get_nodes


@pytest.mark.asyncio
async def test_chunk_id_extracted_from_llm_and_stored(mock_neo4j_vector_calls, mock_llm_with_chunk_id):  # pylint: disable=unused-argument
    """
    Test that chunk_id is extracted from text by the LLM and stored in Neo4j.

    This simulates the complete flow:
    1. Text with chunk_id is sent to LLM
    2. LLM returns nodes with chunk_id in response
    3. chunk_id is stored in Neo4j node properties
    """
    user_id = "test-chunk-id-user"

    async with GraphConstructor(user_id) as constructor:
        # Create user first
        await constructor.graph_ops.create_user(user_id)

        # Create unstructured data (the chunk_id is in the text content)
        data = UnstructuredData(
            title="Understanding Vector Databases",
            content="""
            chunk_id: book_chunk_123

            Vector databases are specialized databases designed to store and query high-dimensional vectors.
            They enable similarity search, which is crucial for modern AI applications like semantic search
            and recommendation systems.
            """,
            chunk_id="book_chunk_123",  # Also passed as metadata
            metadata={
                "book_id": "12345",
                "book_title": "Database Systems",
                "chapter": "7"
            }
        )

        # Ingest the data
        await constructor.ingest_unstructured_data_to_graph(data)

        # Verify nodes were created with chunk_id
        driver = constructor.graph_ops.neo4j_manager.driver
        assert driver is not None

        async with driver.session() as session:
            # Check first node
            result1 = await session.run(
                """
                MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                RETURN n.chunk_id as chunk_id, properties(n) as props
                """,
                node_name="Vector Databases",
                user_id=user_id
            )
            record1 = await result1.single()
            assert record1 is not None, "First node was not created"
            assert record1["chunk_id"] == "book_chunk_123", "chunk_id not stored correctly"

            # Check second node
            result2 = await session.run(
                """
                MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                RETURN n.chunk_id as chunk_id, properties(n) as props
                """,
                node_name="Similarity Search",
                user_id=user_id
            )
            record2 = await result2.single()
            assert record2 is not None, "Second node was not created"
            assert record2["chunk_id"] == "book_chunk_123", "chunk_id not stored correctly"

    # Cleanup
    await constructor.graph_ops.delete_user(user_id)


@pytest.fixture
def mock_llm_without_chunk_id(monkeypatch):
    """Override the autouse fixture to return nodes without chunk_id"""
    mock_llm_nodes = [
        LLMNode(
            name="I love playing guitar",
            type="Preference",
            chunk_id=None,
            discipline=None,
            bloom_level="Remember",
            confidence=0.95
        )
    ]

    async def mock_get_nodes(*args, **kwargs):
        return mock_llm_nodes

    monkeypatch.setattr("persona.llm.llm_graph.get_nodes", mock_get_nodes)
    return mock_get_nodes


@pytest.mark.asyncio
async def test_chunk_id_not_stored_when_not_present(mock_neo4j_vector_calls, mock_llm_without_chunk_id):  # pylint: disable=unused-argument
    """
    Test that chunk_id is NOT stored when it's not provided by the LLM.

    This verifies that nodes created from non-book content (like notes or chat messages)
    don't have a chunk_id property.
    """
    user_id = "test-no-chunk-id-user"

    async with GraphConstructor(user_id) as constructor:
        # Create user first
        await constructor.graph_ops.create_user(user_id)

        # Create unstructured data (personal note without chunk_id)
        data = UnstructuredData(
            title="Personal Note",
            content="I really enjoy playing guitar in my free time. It's very relaxing.",
            metadata={}
        )

        # Ingest the data
        await constructor.ingest_unstructured_data_to_graph(data)

        # Verify node was created without chunk_id
        driver = constructor.graph_ops.neo4j_manager.driver
        assert driver is not None

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                RETURN n.chunk_id as chunk_id, properties(n) as props
                """,
                node_name="I love playing guitar",
                user_id=user_id
            )
            record = await result.single()
            assert record is not None, "Node was not created"

            # chunk_id should be None or not exist
            chunk_id = record["chunk_id"]
            assert chunk_id is None, f"chunk_id should be None but got: {chunk_id}"

            # Verify it's also not in properties
            props = record["props"]
            assert "chunk_id" not in props or props["chunk_id"] is None

    # Cleanup
    await constructor.graph_ops.delete_user(user_id)


@pytest.fixture
def mock_llm_for_ui_test(monkeypatch):
    """Override the autouse fixture for UI test"""
    mock_llm_nodes = [
        LLMNode(
            name="Machine Learning Fundamentals",
            type="Concept",
            chunk_id="book_chunk_456",
            discipline="Computer Science",
            bloom_level="Understand",
            confidence=0.92
        )
    ]

    async def mock_get_nodes(*args, **kwargs):
        return mock_llm_nodes

    monkeypatch.setattr("persona.llm.llm_graph.get_nodes", mock_get_nodes)
    return mock_get_nodes


@pytest.mark.asyncio
async def test_chunk_id_in_graph_ui_response(mock_neo4j_vector_calls, mock_llm_for_ui_test):  # pylint: disable=unused-argument
    """
    Test that chunk_id appears in the graph UI data endpoint response.

    This verifies the complete flow from storage to retrieval via the API.
    """
    user_id = "test-chunk-id-api-user"

    async with GraphConstructor(user_id) as constructor:
        # Create user first
        await constructor.graph_ops.create_user(user_id)

        # Create and ingest data
        data = UnstructuredData(
            title="ML Fundamentals",
            content="chunk_id: book_chunk_456\n\nMachine learning is a subset of AI...",
            chunk_id="book_chunk_456",
            metadata={"book_id": "67890"}
        )

        await constructor.ingest_unstructured_data_to_graph(data)

        # Now test the graph_ui_service retrieval
        from persona.services.graph_ui_service import get_graph_ui_data

        graph_data = await get_graph_ui_data(user_id)

        # Find our node in the response
        node = next((n for n in graph_data["nodes"] if n["name"] == "Machine Learning Fundamentals"), None)
        assert node is not None, "Node not found in graph UI response"

        # Verify chunk_id is in the response
        assert "chunk_id" in node, "chunk_id not in graph UI response"
        assert node["chunk_id"] == "book_chunk_456", f"Expected chunk_id 'book_chunk_456', got {node['chunk_id']}"

        # Verify chunk_id is NOT also in properties (should be top-level only)
        assert "chunk_id" not in node.get("properties", {}), "chunk_id should not be duplicated in properties"

    # Cleanup
    await constructor.graph_ops.delete_user(user_id)
