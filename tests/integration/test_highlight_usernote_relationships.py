"""
Test Suite for Highlight-UserNote ANNOTATED_WITH Relationship Creation

Tests that:
1. LLM-generated ANNOTATED_WITH relationships are filtered out
2. ANNOTATED_WITH relationships are only created by the system
3. Every UserNote has a corresponding Highlight with ANNOTATED_WITH edge
4. Highlights without notes do NOT have ANNOTATED_WITH edges
5. No duplicate ANNOTATED_WITH relationships exist
"""
import pytest
from persona.models.schema import UnstructuredData
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService


TEST_USER_ID = "test_user_highlight_usernote_rels"


@pytest.mark.asyncio
class TestHighlightUserNoteRelationships:
    """Test cases for ANNOTATED_WITH relationship creation"""

    async def test_system_creates_annotated_with_for_highlight_with_note(self):
        """
        Test that ANNOTATED_WITH relationship is automatically created when both
        Highlight and UserNote nodes exist for the same highlight_id.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)
            else:
                # Clean up existing data
                await graph_ops.delete_user(TEST_USER_ID)
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest a highlight with a note
            test_data = UnstructuredData(
                title="Test Highlight with Note",
                content="""
                Highlight: Frank Knight, Henry Simons

                Note: Milton Friedman's advisors at the University of Chicago
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "501",
                    "date": "2025-11-09"
                }
            )

            # Act - Ingest the data
            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Assert - Get all nodes and relationships
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find Highlight and UserNote nodes
            highlights = [n for n in all_nodes if n.type == "Highlight"]
            usernotes = [n for n in all_nodes if n.type == "UserNote"]

            assert len(highlights) == 1, f"Should have 1 Highlight, found {len(highlights)}"
            assert len(usernotes) == 1, f"Should have 1 UserNote, found {len(usernotes)}"

            highlight = highlights[0]
            usernote = usernotes[0]

            # Debug: Print all relationships
            print(f"\nDebug: Total relationships: {len(all_relationships)}")
            rel_types = set(r.relation for r in all_relationships)
            print(f"Debug: Relationship types: {rel_types}")
            for rel in all_relationships:
                print(f"Debug: {rel.source} --[{rel.relation}]--> {rel.target}")

            # Find ANNOTATED_WITH relationship
            annotated_with_rels = [
                rel for rel in all_relationships
                if rel.relation == "ANNOTATED_WITH"
            ]

            assert len(annotated_with_rels) >= 1, \
                f"Should have at least 1 ANNOTATED_WITH relationship, found {len(annotated_with_rels)}"

            rel = annotated_with_rels[0]

            # Verify the relationship connects Highlight -> UserNote
            assert rel.source == highlight.name, \
                f"ANNOTATED_WITH should have Highlight as source, got {rel.source}"
            assert rel.target == usernote.name, \
                f"ANNOTATED_WITH should have UserNote as target, got {rel.target}"

    async def test_no_annotated_with_for_highlight_without_note(self):
        """
        Test that ANNOTATED_WITH relationship is NOT created when only a Highlight
        exists (no UserNote).
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)
            else:
                # Clean up existing data
                await graph_ops.delete_user(TEST_USER_ID)
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest a highlight WITHOUT a note (just reading session)
            test_data = UnstructuredData(
                title="Reading Session: Chapter 5",
                content="""
                This chapter discusses the economic theories of the Chicago School,
                focusing on free market principles and monetarism.
                """,
                metadata={
                    "book_id": "1",
                    "chunk_id": "chapter_5_page_42"
                }
            )

            # Act
            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Assert
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find UserNote nodes (should be 0)
            usernotes = [n for n in all_nodes if n.type == "UserNote"]
            highlights = [n for n in all_nodes if n.type == "Highlight"]

            # This is reading session content, might not have Highlight nodes
            # but definitely should NOT have UserNote nodes
            assert len(usernotes) == 0, f"Should have 0 UserNote nodes, found {len(usernotes)}"

            # Find ANNOTATED_WITH relationships (should be 0)
            annotated_with_rels = [
                rel for rel in all_relationships
                if rel.relation == "ANNOTATED_WITH"
            ]

            assert len(annotated_with_rels) == 0, \
                f"Should have 0 ANNOTATED_WITH relationships when no UserNote exists, found {len(annotated_with_rels)}"

    async def test_llm_cannot_create_annotated_with_relationships(self):
        """
        Test that ANNOTATED_WITH relationships from LLM are filtered out.

        The LLM should NOT be able to create these relationships - they are
        system-managed and created only when highlight_id values match.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)
            else:
                # Clean up existing data
                await graph_ops.delete_user(TEST_USER_ID)
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest data with explicit mention of ANNOTATED_WITH
            # (to potentially trigger LLM to create it)
            test_data = UnstructuredData(
                title="Test with ANNOTATED_WITH mention",
                content="""
                Highlight: The Chicago School of Economics

                Note: This highlight is ANNOTATED_WITH a note about the school's influence.
                The relationship should be system-created, not LLM-created.
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "502",
                    "date": "2025-11-09"
                }
            )

            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get nodes and relationships
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find ANNOTATED_WITH relationships
            annotated_with_rels = [
                rel for rel in all_relationships
                if rel.relation == "ANNOTATED_WITH"
            ]

            # Each ANNOTATED_WITH should connect Highlight to UserNote with matching highlight_id
            highlights = [n for n in all_nodes if n.type == "Highlight"]
            usernotes = [n for n in all_nodes if n.type == "UserNote"]

            assert len(annotated_with_rels) > 0, \
                "System should create ANNOTATED_WITH relationships"

            for rel in annotated_with_rels:
                source_node = next((n for n in all_nodes if n.name == rel.source), None)
                target_node = next((n for n in all_nodes if n.name == rel.target), None)

                assert source_node is not None, \
                    f"Source node '{rel.source}' should exist"
                assert target_node is not None, \
                    f"Target node '{rel.target}' should exist"

                # Verify source is Highlight and target is UserNote
                assert source_node.type == "Highlight", \
                    f"ANNOTATED_WITH source should be Highlight, got {source_node.type}"
                assert target_node.type == "UserNote", \
                    f"ANNOTATED_WITH target should be UserNote, got {target_node.type}"

                # Verify highlight_id matches (this proves it's system-created)
                source_highlight_id = source_node.highlight_id[0] if source_node.highlight_id else None
                target_highlight_id = target_node.highlight_id[0] if target_node.highlight_id else None

                assert source_highlight_id is not None and target_highlight_id is not None, \
                    "Both Highlight and UserNote should have highlight_id"

                assert source_highlight_id == target_highlight_id, \
                    f"Only system can create ANNOTATED_WITH with matching highlight_ids. " \
                    f"Found source={source_highlight_id}, target={target_highlight_id}"

    async def test_no_duplicate_annotated_with_relationships(self):
        """
        Test that no duplicate ANNOTATED_WITH relationships exist.

        Each UserNote should have exactly one ANNOTATED_WITH relationship
        to its corresponding Highlight node (same highlight_id).
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)
            else:
                # Clean up existing data
                await graph_ops.delete_user(TEST_USER_ID)
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest multiple highlights with notes
            test_data_1 = UnstructuredData(
                title="First Highlight",
                content="""
                Highlight: Milton Friedman's monetary theory

                Note: Key insight about money supply and inflation
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "503",
                    "date": "2025-11-09"
                }
            )

            test_data_2 = UnstructuredData(
                title="Second Highlight",
                content="""
                Highlight: The role of government in free markets

                Note: Friedman's view on limited government intervention
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "504",
                    "date": "2025-11-09"
                }
            )

            # Ingest both
            await IngestService.ingest_data(TEST_USER_ID, test_data_1, graph_ops)
            await IngestService.ingest_data(TEST_USER_ID, test_data_2, graph_ops)

            # Get all relationships
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find ANNOTATED_WITH relationships
            annotated_with_rels = [
                rel for rel in all_relationships
                if rel.relation == "ANNOTATED_WITH"
            ]

            # Should have exactly 2 ANNOTATED_WITH relationships (one per highlight/note pair)
            assert len(annotated_with_rels) == 2, \
                f"Should have exactly 2 ANNOTATED_WITH relationships, found {len(annotated_with_rels)}"

            # Check for duplicates: same source and target
            seen_pairs = set()
            duplicates = []

            for rel in annotated_with_rels:
                pair = (rel.source, rel.target)
                if pair in seen_pairs:
                    duplicates.append(pair)
                seen_pairs.add(pair)

            assert len(duplicates) == 0, \
                f"Found duplicate ANNOTATED_WITH relationships: {duplicates}"

    async def test_every_usernote_has_annotated_with_relationship(self):
        """
        Test the core constraint: Every UserNote MUST have an ANNOTATED_WITH
        relationship to a Highlight node.
        """
        async with GraphOps() as graph_ops:
            # Setup
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)
            else:
                # Clean up existing data
                await graph_ops.delete_user(TEST_USER_ID)
                await graph_ops.create_user(TEST_USER_ID)

            # Ingest multiple highlights with notes
            test_data = UnstructuredData(
                title="Multiple Highlights with Notes",
                content="""
                Highlight: Concept A about economics
                Note: Important insight A

                Highlight: Concept B about policy
                Note: Important insight B

                Highlight: Concept C about markets
                Note: Important insight C
                """,
                metadata={
                    "book_id": "1",
                    "highlight_id": "505",
                    "date": "2025-11-09"
                }
            )

            await IngestService.ingest_data(TEST_USER_ID, test_data, graph_ops)

            # Get all nodes and relationships
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Find all UserNote nodes
            usernotes = [n for n in all_nodes if n.type == "UserNote"]

            assert len(usernotes) > 0, "Should have UserNote nodes"

            # For each UserNote, verify it has an ANNOTATED_WITH relationship
            annotated_with_rels = [
                rel for rel in all_relationships
                if rel.relation == "ANNOTATED_WITH"
            ]

            for usernote in usernotes:
                # Find ANNOTATED_WITH relationships pointing to this UserNote
                usernote_rels = [
                    rel for rel in annotated_with_rels
                    if rel.target == usernote.name
                ]

                assert len(usernote_rels) == 1, \
                    f"UserNote '{usernote.name}' should have exactly 1 ANNOTATED_WITH " \
                    f"relationship from a Highlight, found {len(usernote_rels)}"

                # Verify the source is a Highlight
                rel = usernote_rels[0]
                source_node = next((n for n in all_nodes if n.name == rel.source), None)

                assert source_node is not None, \
                    f"Source node '{rel.source}' should exist"
                assert source_node.type == "Highlight", \
                    f"ANNOTATED_WITH source should be Highlight, got {source_node.type}"
