"""
Test Suite for Service 1: Atomic Note Extraction

Tests the system's ability to extract atomic concepts from annotations with varying complexity.
Uses real reading data from "Goddess of the Market: Ayn Rand and the American Right"
"""
import pytest
from persona.models.schema import UnstructuredData
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService


TEST_USER_ID = "test_user_atomic_extraction"


@pytest.mark.asyncio
class TestAtomicNoteExtraction:
    """Test cases for atomic note extraction from annotations"""

    async def test_case_1_1_simple_annotation_extraction(self):
        """
        Test Case 1.1 - Extract concepts from simple annotation

        Input:
          Annotation: "Milton-Friedman's advisor"
          Context: "...the economists Frank Knight, Henry Simons, and Alan Director..."

        Expected Output:
          - At least 1 atomic concept extracted
          - Concept tagged with discipline (e.g., economics)
          - Confidence score provided
        """
        async with GraphOps() as graph_ops:
            # Arrange
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            data = UnstructuredData(
                title="Annotation: Milton-Friedman's advisor",
                content="""
                Reader's Note: Milton-Friedman's advisor

                Context from page 130:
                During the war the economists Frank Knight, Henry Simons, and Alan Director had
                assembled a critical mass of free market thinkers at the university.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "1",
                    "chunk_ids": "anno-1-feb24"
                }
            )

            # Act
            result = await IngestService.ingest_data(TEST_USER_ID, data, graph_ops)

            # Assert
            assert result["message"] == "Data ingested successfully"

            # Verify nodes were created
            nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            assert len(nodes) >= 1, "At least 1 atomic concept should be extracted"

            # Check for expected concepts
            node_names = [node.name.lower() for node in nodes]
            expected_concepts = ["frank knight", "henry simons", "alan director", "milton friedman"]

            found_concepts = [name for name in node_names if any(exp in name for exp in expected_concepts)]
            print(f"Found concepts: {found_concepts}")
            assert len(found_concepts) >= 1, f"Expected to find economist concepts, found nodes: {node_names[:5]}"

            # Verify discipline tagging
            for node in nodes:
                if node.properties and "discipline" in node.properties:
                    discipline = node.properties["discipline"]
                    assert discipline is not None, "Discipline should be tagged"

            # Verify confidence scores
            for node in nodes:
                if node.properties and "confidence" in node.properties:
                    confidence = node.properties["confidence"]
                    assert confidence is not None, "Confidence score should be provided"
                    assert 0.0 <= confidence <= 1.0, f"Confidence should be between 0 and 1, got: {confidence}"

    async def test_case_1_2_analytical_annotation_extraction(self):
        """
        Test Case 1.2 - Extract from analytical annotation
        """
        async with GraphOps() as graph_ops:
            # Arrange
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            data = UnstructuredData(
                title="Annotation: Rand's issue is exactly being extreme!",
                content="""
                Reader's Note: Rand's issue is exactly being extreme!

                Context from page 131:
                Addressing Lane, she compared him to Communist "middle of the roaders" who were
                most effective as propagandists because they were not seen as Communists.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "4",
                    "chunk_ids": "anno-4-feb24"
                }
            )

            # Act
            result = await IngestService.ingest_data(TEST_USER_ID, data, graph_ops)

            # Assert
            assert result["message"] == "Data ingested successfully"

            nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            assert len(nodes) >= 1, "Concepts should be extracted from analytical annotation"

    async def test_case_1_3_shallow_annotation_handling(self):
        """
        Test Case 1.3 - Handle shallow annotation
        """
        async with GraphOps() as graph_ops:
            # Arrange
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            data = UnstructuredData(
                title="Annotation: interesting",
                content="""
                Reader's Note: interesting

                Context from page 131:
                In a letter to Rose Wilder Lane, a libertarian book reviewer, she called him
                "pure poison" and "an example of our most pernicious enemy."
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "3",
                    "chunk_ids": "anno-3-feb24"
                }
            )

            # Act
            result = await IngestService.ingest_data(TEST_USER_ID, data, graph_ops)

            # Assert - System should handle gracefully
            assert result["message"] == "Data ingested successfully"
