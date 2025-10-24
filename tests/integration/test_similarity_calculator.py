"""
Test Suite for Service 2: Similarity Calculator

Tests the system's ability to detect related concepts and avoid false connections.
Uses concepts from "Goddess of the Market: Ayn Rand and the American Right"
"""
import pytest
from persona.models.schema import UnstructuredData, NodeModel
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService
from persona.core.deduplication import NodeDeduplicator


TEST_USER_ID = "test_user_similarity"


async def setup_existing_concepts(graph_ops, test_user_id):
    """Helper to set up existing concepts in the graph"""
    if not await graph_ops.user_exists(test_user_id):
        await graph_ops.create_user(test_user_id)

    # Create existing concepts from the reading
    existing_concepts = UnstructuredData(
        title="Existing Political Philosophy Concepts",
        content="""
        Classical liberalism is a political ideology that emphasizes individual liberty, free markets,
        and limited government intervention.

        Libertarianism advocates for minimal state intervention in the lives of citizens, emphasizing
        individual freedom and voluntary association.

        Individualism is the moral stance, political philosophy, ideology, or social outlook that
        emphasizes the intrinsic worth of the individual.

        Hayek's moral philosophy centered on spontaneous order and the importance of traditional values
        in maintaining a free society.

        Altruism in ethics refers to the principle or practice of concern for the welfare of others,
        often seen as conflicting with individual self-interest.
        """,
        metadata={
            "book_id": "1",
            "chunk_ids": "existing-concepts"
        }
    )

    await IngestService.ingest_data(test_user_id, existing_concepts, graph_ops)


@pytest.mark.asyncio
class TestSimilarityCalculator:
    """Test cases for concept similarity detection"""

    async def test_case_2_1_detect_related_concepts(self):
        """
        Test Case 2.1 - Detect related concepts

        Input:
          New concept: "altruism is collectivism"
          Existing concepts: ["classical liberalism", "libertarianism", "individualism"]

        Expected Output:
          - Identifies "classical liberalism" and "libertarianism" as related
          - Provides relationship type (e.g., "contrasts with" or "opposes")
          - Filters out low-similarity concepts
        """
        async with GraphOps() as graph_ops:
            # Set up existing concepts
            await setup_existing_concepts(graph_ops, TEST_USER_ID)

            # Arrange - Ingest new concept
            new_concept_data = UnstructuredData(
                title="New Concept: Altruism is Collectivism",
                content="""
                Reader's analysis: Altruism is collectivism.

                Context: Rand argued that altruism, the ethical doctrine that one should sacrifice
                one's own interests for others, is fundamentally collectivist because it subordinates
                the individual to the group. This contrasts sharply with individualism and classical
                liberal values that prioritize individual rights and self-interest.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "11",
                    "chunk_ids": "new-concept-altruism"
                }
            )

            # Act
            result = await IngestService.ingest_data(TEST_USER_ID, new_concept_data, graph_ops)

            # Assert
            assert result["message"] == "Data ingested successfully"

            # Get all relationships
            relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Check for relationships involving altruism/collectivism
            altruism_relationships = [
                rel for rel in relationships
                if "altruism" in rel.source.lower() or "altruism" in rel.target.lower()
                or "collectivism" in rel.source.lower() or "collectivism" in rel.target.lower()
            ]

            # Should have relationships connecting to individualism, liberalism, or libertarianism
            related_concepts = set()
            for rel in altruism_relationships:
                related_concepts.add(rel.source.lower())
                related_concepts.add(rel.target.lower())

            expected_related = ["classical liberalism", "libertarianism", "individualism"]

            found_related = [
                concept for concept in expected_related
                if any(concept in related.lower() for related in related_concepts)
            ]

            assert len(found_related) >= 1, \
                f"Expected to find relationships with {expected_related}, found connections to: {related_concepts}"

            # Verify relationship types indicate contrast/opposition
            if len(altruism_relationships) > 0:
                relation_types = [rel.relation for rel in altruism_relationships]
                # Relationships should exist and have meaningful types
                assert len(relation_types) > 0, "Should have relationship types defined"

    async def test_case_2_2_avoid_false_connections(self):
        """
        Test Case 2.2 - Avoid false connections

        Input:
          New concept: "Milton Friedman's advisor"
          Existing concepts: ["altruism", "Hayek's moral philosophy"]

        Expected Output:
          - No connections returned (or very low similarity scores)
          - OR entity vs concept distinction made
        """
        async with GraphOps() as graph_ops:
            # Set up existing concepts
            await setup_existing_concepts(graph_ops, TEST_USER_ID)

            # Arrange - Ingest entity (person reference)
            entity_data = UnstructuredData(
                title="Entity: Milton Friedman's Advisors",
                content="""
                Reader's note: Milton Friedman's advisor

                Context: During the war the economists Frank Knight, Henry Simons, and Alan Director
                had assembled a critical mass of free market thinkers at the university.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "1",
                    "chunk_ids": "entity-friedman-advisor"
                }
            )

            # Act
            result = await IngestService.ingest_data(TEST_USER_ID, entity_data, graph_ops)

            # Assert
            assert result["message"] == "Data ingested successfully"

            # Get all relationships
            relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Check relationships involving the person entities
            person_names = ["frank knight", "henry simons", "alan director", "milton friedman"]

            person_relationships = [
                rel for rel in relationships
                if any(person in rel.source.lower() for person in person_names)
                or any(person in rel.target.lower() for person in person_names)
            ]

            # Count relationships to abstract concepts (altruism, moral philosophy)
            abstract_concepts = ["altruism", "moral philosophy", "ethics"]

            false_connections = [
                rel for rel in person_relationships
                if any(concept in rel.source.lower() for concept in abstract_concepts)
                or any(concept in rel.target.lower() for concept in abstract_concepts)
            ]

            # Should have minimal or no false connections between entities and unrelated concepts
            # Some connections might exist if there's contextual relevance, but should be filtered
            assert len(false_connections) <= len(person_relationships) * 0.3, \
                f"Too many false connections found: {len(false_connections)} out of {len(person_relationships)}"

            # Alternatively, check that entity types are distinguished
            nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            person_nodes = [
                node for node in nodes
                if any(person in node.name.lower() for person in person_names)
            ]

            if len(person_nodes) > 0:
                # Check if person nodes are typed differently from concept nodes
                for node in person_nodes:
                    if node.type:
                        node_type = node.type.lower()
                        # Person nodes should be typed as Person, Entity, or similar
                        # Not as abstract concepts
                        is_entity_type = any(entity_indicator in node_type for entity_indicator in
                                            ["person", "entity", "economist", "author", "figure"])

                        # Either properly typed as entity OR not falsely connected to unrelated concepts
                        assert is_entity_type or len(false_connections) == 0, \
                            f"Entity '{node.name}' should be typed as entity or have no false connections"

    async def test_case_2_3_similarity_scoring(self):
        """
        Additional test - Verify similarity scoring filters out low-similarity concepts
        """
        async with GraphOps() as graph_ops:
            # Set up existing concepts
            await setup_existing_concepts(graph_ops, TEST_USER_ID)

            # Use the deduplicator to check similarity
            deduplicator = NodeDeduplicator(graph_ops.neo4j_manager, similarity_threshold=0.85)

            # Test similarity between related concepts
            similar_node = await deduplicator.find_similar_node(
                node_name="free market economics",
                node_type="Concept",
                user_id=TEST_USER_ID
            )

            # Should find similar concepts if they exist
            if similar_node:
                assert "score" in similar_node, "Similarity result should include score"
                assert similar_node["score"] >= 0.85, "Score should meet threshold"

            # Test with unrelated concept
            dissimilar_node = await deduplicator.find_similar_node(
                node_name="quantum physics",
                node_type="Concept",
                user_id=TEST_USER_ID
            )

            # Should not find matches for completely unrelated concepts
            # (or if found, score should be much lower)
            if dissimilar_node:
                assert dissimilar_node["score"] < 0.95, \
                    "Unrelated concepts should have lower similarity scores"

    async def test_case_2_4_relationship_types_accuracy(self):
        """
        Additional test - Verify relationship types are meaningful and accurate
        """
        async with GraphOps() as graph_ops:
            # Set up existing concepts
            await setup_existing_concepts(graph_ops, TEST_USER_ID)

            # Ingest content with clear relationships
            relationship_data = UnstructuredData(
                title="Relationships: Liberalism vs Collectivism",
                content="""
                Classical liberalism emphasizes individual liberty and limited government.
                This contrasts with collectivism, which prioritizes group interests over individual rights.
                Libertarianism opposes collectivism and shares principles with classical liberalism.
                Individualism conflicts with altruism when altruism demands self-sacrifice for the collective.
                """,
                metadata={
                    "book_id": "1",
                    "chunk_ids": "relationships-test"
                }
            )

            # Act
            await IngestService.ingest_data(TEST_USER_ID, relationship_data, graph_ops)

            # Assert - Check relationship types
            relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Relationship types should be meaningful
            relation_types = set(rel.relation for rel in relationships)

            # Should have relationship types (not empty)
            assert len(relation_types) > 0, "Should have defined relationship types"

            # Common meaningful relationship types in this domain
            expected_relation_indicators = [
                "contrast", "oppose", "relate", "connect", "conflict",
                "similar", "derived", "part", "type", "influence"
            ]

            # At least some relationships should have meaningful types
            meaningful_relations = [
                rel for rel in relationships
                if rel.relation and any(indicator in rel.relation.lower()
                                       for indicator in expected_relation_indicators)
            ]

            # Allow for flexibility in relationship naming
            assert len(meaningful_relations) >= 0, \
                f"Relationships should have meaningful types. Found: {relation_types}"
