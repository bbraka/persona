"""
Integration test for contrast/semantic relationship detection.

This test verifies that the system can detect philosophical opposites and contrasting
concepts that vector similarity alone would miss.

Test Case:
- New concept: "altruism is collectivism"
- Existing concepts: "classical liberalism", "libertarianism", "individualism"
- Expected: System detects these as contrasting relationships (CONTRASTS_WITH, OPPOSES)
"""

import pytest
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService
from persona.models.schema import UnstructuredData

TEST_USER_ID = "test_user_contrast_detection"
TEST_BOOK_ID = 99999


@pytest.mark.asyncio
class TestContrastDetection:
    """Test cases for detecting contrasting and opposing concept relationships"""

    async def test_detect_contrasting_concepts(self):
        """
        Test that the system detects philosophical opposites/contrasts.

        Scenario:
        1. Ingest existing concepts about individualism/liberalism
        2. Ingest new concept about collectivism/altruism
        3. Verify system creates CONTRASTS_WITH or OPPOSES relationships
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Clean existing test data
            await graph_ops.clean_graph()
            await graph_ops.create_user(TEST_USER_ID)
            # Recreate vector index after clean_graph()
            await graph_ops.neo4j_manager.ensure_vector_index()

            # Step 1: Ingest existing concepts (individualism, liberalism, etc.)
            existing_concepts = UnstructuredData(
                title="Political Philosophy Concepts",
                content="""
                HIGHLIGHT: "classical liberalism emphasizes individual rights and limited government"

                Classical liberalism is a political philosophy that values individual liberty,
                free markets, and limited government intervention.

                HIGHLIGHT: "libertarianism prioritizes individual freedom above all"

                Libertarianism extends classical liberal principles to advocate for minimal
                state intervention in both economic and personal matters.

                HIGHLIGHT: "individualism emphasizes personal autonomy and self-determination"

                Individualism is the moral stance that prioritizes the individual over the collective,
                emphasizing personal freedom, self-reliance, and individual rights.
                """,
                metadata={
                    "user_id": TEST_USER_ID,
                    "book_id": str(TEST_BOOK_ID),
                    "book_title": "Political Philosophy",
                    "book_author": "Test Author",
                    "chapter": "Chapter 1"
                }
            )

            result1 = await IngestService.ingest_data(
                user_id=TEST_USER_ID,
                data=existing_concepts,
                graph_ops=graph_ops
            )

            assert "message" in result1

            # Step 2: Ingest contrasting concept (collectivism/altruism)
            contrasting_concept = UnstructuredData(
                title="Collectivism and Altruism",
                content="""
                HIGHLIGHT: "altruism is collectivism - it prioritizes the group over the individual"

                Altruism is fundamentally collectivist because it demands that individuals
                sacrifice their own interests for the sake of others or society as a whole.
                This stands in direct opposition to individualist philosophies.
                """,
                metadata={
                    "user_id": TEST_USER_ID,
                    "book_id": str(TEST_BOOK_ID),
                    "book_title": "Political Philosophy",
                    "book_author": "Test Author",
                    "chapter": "Chapter 2"
                }
            )

            result2 = await IngestService.ingest_data(
                user_id=TEST_USER_ID,
                data=contrasting_concept,
                graph_ops=graph_ops
            )

            assert "message" in result2

            # Step 3: Verify relationships were created
            all_rels = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Filter for relationships involving the new collectivism concept
            collectivism_rels = [
                rel for rel in all_rels
                if "altruism" in rel.source.lower() or "collectivism" in rel.source.lower()
            ]

            # Check that we found contrasting relationships
            assert len(collectivism_rels) > 0, "Should have created relationships from the altruism/collectivism concept"

            # Look for specific relationship types indicating contrast/opposition
            contrast_types = ["CONTRASTS_WITH", "OPPOSES", "ARGUES_AGAINST", "CHALLENGES"]
            opposing_rels = [
                rel for rel in collectivism_rels
                if rel.relation in contrast_types
            ]

            # Verify we found at least one contrasting relationship
            assert len(opposing_rels) > 0, (
                f"Should have detected at least one contrasting relationship. "
                f"Found relationships: {[(r.source[:50], r.relation, r.target[:50]) for r in collectivism_rels]}"
            )

            # Verify targets include the expected concepts
            targets = [rel.target.lower() for rel in collectivism_rels]
            expected_targets = ["individualism", "libertarianism", "classical liberalism"]

            found_targets = []
            for expected in expected_targets:
                if any(expected in target for target in targets):
                    found_targets.append(expected)

            assert len(found_targets) >= 1, (
                f"Should have connected to at least one of {expected_targets}. "
                f"Found targets: {targets}"
            )

            print(f"\n✓ Successfully detected {len(opposing_rels)} contrasting relationships")
            print(f"✓ Connected to: {', '.join(found_targets)}")
            for rel in opposing_rels[:3]:  # Show first 3
                print(f"  - {rel.source[:60]} {rel.relation} {rel.target[:60]}")

    async def test_contrast_detection_with_same_discipline(self):
        """
        Test that concepts in the same discipline are more likely to be connected,
        even with lower vector similarity.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            # Clean and recreate
            await graph_ops.clean_graph()
            await graph_ops.create_user(TEST_USER_ID)
            # Recreate vector index after clean_graph()
            await graph_ops.neo4j_manager.ensure_vector_index()

            # Ingest concepts from same discipline (Economics) with opposing views
            keynesian = UnstructuredData(
                title="Keynesian Economics",
                content="""
                HIGHLIGHT: "keynesian economics advocates government intervention to manage economic cycles"

                Keynesian economics supports active fiscal policy and government spending
                to smooth out business cycles and maintain full employment.
                """,
                metadata={
                    "user_id": TEST_USER_ID,
                    "book_id": str(TEST_BOOK_ID),
                    "book_title": "Economic Theories",
                    "book_author": "Test Author",
                    "chapter": "Chapter 1"
                }
            )

            result1 = await IngestService.ingest_data(
                user_id=TEST_USER_ID,
                data=keynesian,
                graph_ops=graph_ops
            )

            assert "message" in result1

            # Ingest opposing view
            austrian = UnstructuredData(
                title="Austrian Economics",
                content="""
                HIGHLIGHT: "austrian economics opposes government intervention and advocates free markets"

                Austrian economics rejects government intervention in the economy,
                emphasizing individual choice, free markets, and the impossibility of central planning.
                """,
                metadata={
                    "user_id": TEST_USER_ID,
                    "book_id": str(TEST_BOOK_ID),
                    "book_title": "Economic Theories",
                    "book_author": "Test Author",
                    "chapter": "Chapter 2"
                }
            )

            result2 = await IngestService.ingest_data(
                user_id=TEST_USER_ID,
                data=austrian,
                graph_ops=graph_ops
            )

            assert "message" in result2

            # Verify opposing relationship was created
            all_rels = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find relationships between the two economic theories
            econ_rels = [
                rel for rel in all_rels
                if ("austrian" in rel.source.lower() and "keynesian" in rel.target.lower()) or
                   ("keynesian" in rel.source.lower() and "austrian" in rel.target.lower())
            ]

            assert len(econ_rels) > 0, (
                "Should have connected Keynesian and Austrian economics despite different terminology. "
                f"Total relationships: {len(all_rels)}"
            )

            print(f"\n✓ Successfully connected opposing economic theories")
            for rel in econ_rels:
                print(f"  - {rel.source[:60]} {rel.relation} {rel.target[:60]}")
