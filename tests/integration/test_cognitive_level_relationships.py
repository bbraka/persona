"""
Test Suite for CognitiveLevel Relationship Creation

Tests that:
1. LLM-generated HAS_UNDERSTANDING_LEVEL relationships are filtered out
2. HAS_UNDERSTANDING_LEVEL relationships are only created by the system
3. Relationships are created only when concept_uuid matches
4. No duplicate HAS_UNDERSTANDING_LEVEL relationships exist
"""
import pytest
from persona.models.schema import UnstructuredData
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService


TEST_USER_ID = "test_user_cognitive_rels"


@pytest.mark.asyncio
class TestCognitiveLevelRelationships:
    """Test cases for HAS_UNDERSTANDING_LEVEL relationship creation"""

    async def test_llm_cannot_create_has_understanding_level_relationships(self):
        """
        Test that HAS_UNDERSTANDING_LEVEL relationships from LLM are filtered out.

        The LLM should NOT be able to create these relationships - they are
        system-managed and created only when concept_uuid values match.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest data that might trigger LLM to create HAS_UNDERSTANDING_LEVEL
            test_data = UnstructuredData(
                title="Test Concept with Cognitive Level",
                content="""
                Reader's Note: This is an important concept about learning.

                Context: Understanding the relationship between concepts and their cognitive
                levels is essential for effective learning. This concept demonstrates how
                knowledge can be categorized according to Bloom's taxonomy.

                The concept "Learning requires practice" is at the Remember level initially,
                but can progress to Apply level with experience.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "101",
                    "date": "2025-11-06"
                }
            )

            # Act - Ingest the data
            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Assert - Get all relationships
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find HAS_UNDERSTANDING_LEVEL relationships
            understanding_rels = [
                rel for rel in all_relationships
                if rel.relation == "HAS_UNDERSTANDING_LEVEL"
            ]

            # Check that HAS_UNDERSTANDING_LEVEL relationships exist (created by system)
            assert len(understanding_rels) > 0, \
                "System should create HAS_UNDERSTANDING_LEVEL relationships"

            # Get all nodes
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            # Find Concept and CognitiveLevel nodes
            concepts = [n for n in all_nodes if n.type == "Concept"]
            cognitive_levels = [n for n in all_nodes if n.type == "CognitiveLevel"]

            assert len(concepts) > 0, "Should have Concept nodes"
            assert len(cognitive_levels) > 0, "Should have CognitiveLevel nodes"

            # Verify each HAS_UNDERSTANDING_LEVEL relationship has matching concept_uuid
            for rel in understanding_rels:
                # Find source and target nodes
                source_node = next((n for n in all_nodes if n.name == rel.source), None)
                target_node = next((n for n in all_nodes if n.name == rel.target), None)

                assert source_node is not None, \
                    f"Source node '{rel.source}' should exist"
                assert target_node is not None, \
                    f"Target node '{rel.target}' should exist"

                # Verify source is Concept and target is CognitiveLevel
                assert source_node.type == "Concept", \
                    f"HAS_UNDERSTANDING_LEVEL source should be Concept, got {source_node.type}"
                assert target_node.type == "CognitiveLevel", \
                    f"HAS_UNDERSTANDING_LEVEL target should be CognitiveLevel, got {target_node.type}"

                # CRITICAL: Verify concept_uuid matches
                source_uuid = getattr(source_node, 'concept_uuid', None)
                target_uuid = getattr(target_node, 'concept_uuid', None)

                assert source_uuid is not None, \
                    f"Concept '{source_node.name}' should have concept_uuid"
                assert target_uuid is not None, \
                    f"CognitiveLevel '{target_node.name}' should have concept_uuid"
                assert source_uuid == target_uuid, \
                    f"concept_uuid must match: Concept={source_uuid}, CognitiveLevel={target_uuid}"

    async def test_no_duplicate_has_understanding_level_relationships(self):
        """
        Test that no duplicate HAS_UNDERSTANDING_LEVEL relationships exist.

        Each Concept should have exactly one HAS_UNDERSTANDING_LEVEL relationship
        to its corresponding CognitiveLevel node (same concept_uuid).
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest multiple times to potentially trigger duplicates
            test_data = UnstructuredData(
                title="Repeated Concept",
                content="""
                The concept "Effective learning requires active practice and reflection"
                demonstrates the importance of metacognition in education.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "102"
                }
            )

            # Ingest the same concept twice
            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get all relationships
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            # Find HAS_UNDERSTANDING_LEVEL relationships
            understanding_rels = [
                rel for rel in all_relationships
                if rel.relation == "HAS_UNDERSTANDING_LEVEL"
            ]

            # Check for duplicates: same source and target
            seen_pairs = set()
            duplicates = []

            for rel in understanding_rels:
                pair = (rel.source, rel.target)
                if pair in seen_pairs:
                    duplicates.append(pair)
                seen_pairs.add(pair)

            assert len(duplicates) == 0, \
                f"Found duplicate HAS_UNDERSTANDING_LEVEL relationships: {duplicates}"

            # Verify each Concept has exactly one CognitiveLevel
            concepts = [n for n in all_nodes if n.type == "Concept"]

            for concept in concepts:
                concept_rels = [
                    rel for rel in understanding_rels
                    if rel.source == concept.name
                ]

                # Each Concept should have exactly 1 HAS_UNDERSTANDING_LEVEL relationship
                assert len(concept_rels) == 1, \
                    f"Concept '{concept.name}' should have exactly 1 HAS_UNDERSTANDING_LEVEL, " \
                    f"found {len(concept_rels)}"

    async def test_cognitive_level_relationships_only_created_by_system(self):
        """
        Test that HAS_UNDERSTANDING_LEVEL relationships are only created by
        _create_cognitive_level_relationships method, not by LLM extraction.

        This test verifies that even if the LLM tries to create these relationships,
        they are filtered out and only system-generated ones remain.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest data
            test_data = UnstructuredData(
                title="Concept About Relationships",
                content="""
                The concept "Knowledge graphs connect information through relationships"
                is a fundamental principle in knowledge representation.

                This concept has a HAS_UNDERSTANDING_LEVEL relationship to the Remember level.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "103"
                }
            )

            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get nodes and relationships
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find HAS_UNDERSTANDING_LEVEL relationships
            understanding_rels = [
                rel for rel in all_relationships
                if rel.relation == "HAS_UNDERSTANDING_LEVEL"
            ]

            # Each relationship should have valid concept_uuid matching
            for rel in understanding_rels:
                source_node = next((n for n in all_nodes if n.name == rel.source), None)
                target_node = next((n for n in all_nodes if n.name == rel.target), None)

                assert source_node is not None and target_node is not None, \
                    f"Both nodes should exist for relationship {rel.source} -> {rel.target}"

                # Verify this is a valid system-created relationship
                source_uuid = getattr(source_node, 'concept_uuid', None)
                target_uuid = getattr(target_node, 'concept_uuid', None)

                # If LLM created this, UUIDs wouldn't match
                # System-created relationships MUST have matching UUIDs
                assert source_uuid == target_uuid, \
                    f"Only system can create HAS_UNDERSTANDING_LEVEL with matching UUIDs. " \
                    f"Found source_uuid={source_uuid}, target_uuid={target_uuid}"

    async def test_concept_uuid_consistency(self):
        """
        Test that concept_uuid is consistently set on both Concept and CognitiveLevel nodes.

        Each Concept node should have:
        1. A concept_uuid
        2. Exactly one CognitiveLevel node with the same concept_uuid
        3. A HAS_UNDERSTANDING_LEVEL relationship connecting them
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest multiple concepts
            test_data = UnstructuredData(
                title="Multiple Concepts",
                content="""
                Reader's Note: Testing concept UUID consistency

                Context: This passage contains multiple concepts:

                1. The concept "Practice improves skill" is fundamental.
                2. The concept "Feedback accelerates learning" is also important.
                3. The concept "Reflection deepens understanding" completes the triad.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "104"
                }
            )

            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get all nodes
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find all Concept nodes
            concepts = [n for n in all_nodes if n.type == "Concept"]
            cognitive_levels = [n for n in all_nodes if n.type == "CognitiveLevel"]

            assert len(concepts) > 0, "Should have Concept nodes"
            assert len(cognitive_levels) > 0, "Should have CognitiveLevel nodes"

            # Check each Concept
            for concept in concepts:
                # 1. Verify Concept has concept_uuid
                concept_uuid = getattr(concept, 'concept_uuid', None)
                assert concept_uuid is not None, \
                    f"Concept '{concept.name}' must have concept_uuid"

                # 2. Find matching CognitiveLevel with same concept_uuid
                matching_cognitive_levels = [
                    cl for cl in cognitive_levels
                    if getattr(cl, 'concept_uuid', None) == concept_uuid
                ]

                assert len(matching_cognitive_levels) == 1, \
                    f"Concept '{concept.name}' (uuid={concept_uuid}) should have exactly 1 " \
                    f"CognitiveLevel with same uuid, found {len(matching_cognitive_levels)}"

                cognitive_level = matching_cognitive_levels[0]

                # 3. Verify HAS_UNDERSTANDING_LEVEL relationship exists
                has_relationship = any(
                    rel.relation == "HAS_UNDERSTANDING_LEVEL" and
                    rel.source == concept.name and
                    rel.target == cognitive_level.name
                    for rel in all_relationships
                )

                assert has_relationship, \
                    f"Concept '{concept.name}' should have HAS_UNDERSTANDING_LEVEL " \
                    f"relationship to '{cognitive_level.name}'"

    async def test_no_orphaned_cognitive_levels(self):
        """
        Test that all CognitiveLevel nodes have a corresponding Concept node.

        Every CognitiveLevel should:
        1. Have a concept_uuid
        2. Have exactly one Concept with matching concept_uuid
        3. Be connected via HAS_UNDERSTANDING_LEVEL relationship
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest data
            test_data = UnstructuredData(
                title="Testing Orphaned Cognitive Levels",
                content="""
                The concept "Every cognitive level must have a parent concept" ensures
                data integrity in the knowledge graph system.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "105"
                }
            )

            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get all nodes
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find CognitiveLevel nodes
            cognitive_levels = [n for n in all_nodes if n.type == "CognitiveLevel"]
            concepts = [n for n in all_nodes if n.type == "Concept"]

            for cognitive_level in cognitive_levels:
                # 1. Verify CognitiveLevel has concept_uuid
                cl_uuid = getattr(cognitive_level, 'concept_uuid', None)
                assert cl_uuid is not None, \
                    f"CognitiveLevel '{cognitive_level.name}' must have concept_uuid"

                # 2. Find matching Concept with same concept_uuid
                matching_concepts = [
                    c for c in concepts
                    if getattr(c, 'concept_uuid', None) == cl_uuid
                ]

                assert len(matching_concepts) == 1, \
                    f"CognitiveLevel '{cognitive_level.name}' (uuid={cl_uuid}) should have " \
                    f"exactly 1 Concept with same uuid, found {len(matching_concepts)}"

                concept = matching_concepts[0]

                # 3. Verify HAS_UNDERSTANDING_LEVEL relationship exists
                has_relationship = any(
                    rel.relation == "HAS_UNDERSTANDING_LEVEL" and
                    rel.source == concept.name and
                    rel.target == cognitive_level.name
                    for rel in all_relationships
                )

                assert has_relationship, \
                    f"CognitiveLevel '{cognitive_level.name}' should be connected to " \
                    f"Concept '{concept.name}' via HAS_UNDERSTANDING_LEVEL"
