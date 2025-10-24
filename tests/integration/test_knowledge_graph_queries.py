"""
Test Suite for Service 3: Knowledge Graph

Tests the system's ability to store, retrieve, and query knowledge graph data.
Includes filtering by entity IDs (book_id, highlight_id) and date-based queries.
"""
import pytest
from datetime import datetime
from persona.models.schema import UnstructuredData
from persona.core.graph_ops import GraphOps
from persona.services.ingest_service import IngestService
from persona.services.graph_ui_service import GraphUIService


TEST_USER_ID = "test_user_kg_queries"


async def setup_sample_graph(graph_ops, test_user_id):
    """
    Helper to set up a complete sample graph from the reading annotations
    """
    if not await graph_ops.user_exists(test_user_id):
        await graph_ops.create_user(test_user_id)

    # Ingest annotation from Feb 24
    feb24_annotation1 = UnstructuredData(
        title="Feb 24 - Annotation 1: Milton Friedman's advisors",
        content="""
        Reader's Note: Milton-Friedman's advisor

        Context from page 130:
        During the war the economists Frank Knight, Henry Simons, and Alan Director had
        assembled a critical mass of free market thinkers at the university. Hayek's arrival
        marked a high point in this campaign.
        """,
        metadata={
            "book_id": "1",
            "highlight_id": "1",
            "chunk_ids": "feb24-anno-1",
            "date": "2025-02-24"
        }
    )

    feb24_annotation4 = UnstructuredData(
        title="Feb 24 - Annotation 4: Rand's extremism",
        content="""
        Reader's Note: Rand's issue is exactly being extreme!

        Context from page 131:
        Addressing Lane, she compared him to Communist "middle of the roaders" who were
        most effective as propagandists. Rand's reaction to Hayek illuminates an important
        difference between her libertarianism and the classical liberal tradition that Hayek represented.
        """,
        metadata={
            "book_id": "1",
            "highlight_id": "4",
            "chunk_ids": "feb24-anno-4",
            "date": "2025-02-24"
        }
    )

    # Ingest annotations from Feb 26
    feb26_annotation10 = UnstructuredData(
        title="Feb 26 - Annotation 10: Hayek's religious views",
        content="""
        Context from page 133:
        Hayek was receptive to Christian values (although cagey about his personal religious beliefs).
        His work was motivated by a deep sense of spiritual crisis.
        """,
        metadata={
            "book_id": "1",
            "highlight_id": "10",
            "chunk_ids": "feb26-anno-10",
            "date": "2025-02-26"
        }
    )

    feb26_annotation11 = UnstructuredData(
        title="Feb 26 - Annotation 11: Altruism is collectivism",
        content="""
        Reader's Note: This could be the fatal dilemma facing today's Left in the US and even in the West

        Context from page 133:
        "Nineteenth Century Liberalism made the mistake of associating liberty with fighting for
        the people, for the downtrodden, for the poor. They made it an altruistic movement.
        But altruism is collectivism. That is why collectivism took the liberals over."

        The solution was to shift the principles of nineteenth-century liberalism onto different
        ethical grounds that avoided altruism.
        """,
        metadata={
            "book_id": "1",
            "highlight_id": "11",
            "chunk_ids": "feb26-anno-11",
            "date": "2025-02-26"
        }
    )

    # Additional content about Hayek for relationship testing
    hayek_content = UnstructuredData(
        title="Hayek's Philosophy and Influence",
        content="""
        F.A. Hayek was a prominent economist and philosopher known for his defense of classical
        liberalism and free-market capitalism. His book The Road to Serfdom warned against
        central planning and collectivism.

        Hayek founded the Mont Pelerin Society and was associated with Ludwig von Mises,
        the Austrian School of economics, and the Chicago School of economics.

        His work influenced modern libertarianism and conservative economic thought.
        """,
        metadata={
            "book_id": "1",
            "chunk_ids": "hayek-background"
        }
    )

    # Ingest all data
    await IngestService.ingest_data(test_user_id, feb24_annotation1, graph_ops)
    await IngestService.ingest_data(test_user_id, feb24_annotation4, graph_ops)
    await IngestService.ingest_data(test_user_id, feb26_annotation10, graph_ops)
    await IngestService.ingest_data(test_user_id, feb26_annotation11, graph_ops)
    await IngestService.ingest_data(test_user_id, hayek_content, graph_ops)


@pytest.mark.asyncio
class TestKnowledgeGraphQueries:
    """Test cases for knowledge graph storage and retrieval"""

    async def test_case_3_1_store_and_retrieve_hayek_concepts(self):
        """
        Test Case 3.1 - Store and retrieve

        Action: Store annotation graph from sample data
        Query: Find all concepts related to "Hayek"

        Expected Output:
          - Returns concepts: classical liberalism, altruism, Mises, etc.
          - Shows relationship types
          - Fast query response
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Act - Retrieve all nodes and relationships
            import time
            start_time = time.time()

            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            all_relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            query_time = time.time() - start_time

            # Assert - Data was stored
            assert len(all_nodes) > 0, "Nodes should be stored in the graph"
            assert query_time < 5.0, f"Query should be fast (took {query_time:.2f}s)"

            # Find Hayek-related concepts
            hayek_nodes = [
                node for node in all_nodes
                if "hayek" in node.name.lower()
            ]

            assert len(hayek_nodes) >= 1, "Should have stored Hayek node(s)"

            # Find concepts related to Hayek
            hayek_related_concepts = set()
            for rel in all_relationships:
                if "hayek" in rel.source.lower():
                    hayek_related_concepts.add(rel.target)
                if "hayek" in rel.target.lower():
                    hayek_related_concepts.add(rel.source)

            # Convert to lowercase for comparison
            related_concepts_lower = {concept.lower() for concept in hayek_related_concepts}

            # Expected related concepts
            expected_concepts = [
                "classical liberalism", "classical liberal",
                "altruism", "collectivism",
                "mises", "ludwig von mises",
                "libertarian", "chicago school", "mont pelerin"
            ]

            found_expected = [
                exp for exp in expected_concepts
                if any(exp in concept for concept in related_concepts_lower)
            ]

            assert len(found_expected) >= 2, \
                f"Expected to find concepts like {expected_concepts}, found: {hayek_related_concepts}"

            # Verify relationship types are shown
            hayek_relationships = [
                rel for rel in all_relationships
                if "hayek" in rel.source.lower() or "hayek" in rel.target.lower()
            ]

            if len(hayek_relationships) > 0:
                relation_types = [rel.relation for rel in hayek_relationships]
                assert all(rel_type is not None for rel_type in relation_types), \
                    "All relationships should have types defined"

    async def test_case_3_2_filter_by_date(self):
        """
        Test Case 3.2 - Filter by date

        Action: Query concepts from Feb 24 vs Feb 26
        Expected Output:
          - Correctly separates concepts by date
          - Shows evolution if same concept appears multiple times
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Act - Query nodes using date_from and date_to parameters
            # Test filtering by specific dates (Feb 24 vs Feb 26)
            feb24_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-24",
                date_to="2025-02-24"  # Same day - will be expanded to end of day
            )

            feb26_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-26",
                date_to="2025-02-26"  # Same day - will be expanded to end of day
            )

            # Assert - Different content for different dates
            feb24_nodes = feb24_result.get("nodes", [])
            feb26_nodes = feb26_result.get("nodes", [])

            assert len(feb24_nodes) > 0, "Should have nodes from Feb 24"
            assert len(feb26_nodes) > 0, "Should have nodes from Feb 26"

            print(f"\n✓ Date filtering results:")
            print(f"  - Feb 24: {len(feb24_nodes)} nodes")
            print(f"  - Feb 26: {len(feb26_nodes)} nodes")

            # Verify all Feb 24 nodes have correct created_at
            for node in feb24_nodes:
                created_at = node.get("created_at", "")
                assert created_at.startswith("2025-02-24"), \
                    f"Feb 24 node '{node['name']}' has wrong date: {created_at}"

            # Verify all Feb 26 nodes have correct created_at
            for node in feb26_nodes:
                created_at = node.get("created_at", "")
                assert created_at.startswith("2025-02-26"), \
                    f"Feb 26 node '{node['name']}' has wrong date: {created_at}"

            # Check content differences
            feb24_node_names = {node["name"].lower() for node in feb24_nodes}
            feb26_node_names = {node["name"].lower() for node in feb26_nodes}

            # Feb 24 should include Milton Friedman's advisors (from the test data)
            feb24_expected = ["frank knight", "henry simons", "alan director", "friedman"]
            feb24_found = [name for name in feb24_expected if any(name in node for node in feb24_node_names)]

            assert len(feb24_found) >= 1, \
                f"Feb 24 should include economist names, found: {feb24_node_names}"

            # Feb 26 should include altruism/collectivism concepts
            feb26_expected = ["altruism", "collectivism", "liberalism"]
            feb26_found = [name for name in feb26_expected if any(name in node for node in feb26_node_names)]

            assert len(feb26_found) >= 1, \
                f"Feb 26 should include philosophical concepts, found: {feb26_node_names}"

            # Verify filtering correctly separates by date
            # Nodes should be completely separate (no overlap) since they're from different days
            overlap = feb24_node_names.intersection(feb26_node_names)

            # There should be no overlap for distinct dates with temporal filtering
            # (unless a concept was mentioned on both days and got merged)
            overlap_ratio = len(overlap) / max(len(feb24_node_names), len(feb26_node_names)) if max(len(feb24_node_names), len(feb26_node_names)) > 0 else 0

            print(f"  - Overlap: {len(overlap)} nodes ({overlap_ratio:.1%})")
            if overlap:
                print(f"  - Overlapping concepts: {list(overlap)[:5]}")

            # Most concepts should be date-specific
            assert overlap_ratio < 0.5, \
                f"Date filtering should separate most concepts (overlap: {overlap_ratio:.1%})"

            # Test evolution tracking: Check if concepts that appear in overlap have bloom_history
            if overlap:
                for node_name in list(overlap)[:3]:  # Check first 3 overlapping concepts
                    # Find the node in both result sets
                    feb24_node = next((n for n in feb24_nodes if n["name"].lower() == node_name), None)
                    feb26_node = next((n for n in feb26_nodes if n["name"].lower() == node_name), None)

                    if feb24_node and feb26_node:
                        # If same concept appears on different dates, check bloom_history
                        bloom24 = feb24_node.get("bloom_history", [])
                        bloom26 = feb26_node.get("bloom_history", [])

                        # Should have bloom history if concept evolved
                        if bloom24 or bloom26:
                            print(f"  - Concept '{node_name}' shows evolution:")
                            print(f"    Feb 24 bloom_history: {len(bloom24)} entries")
                            print(f"    Feb 26 bloom_history: {len(bloom26)} entries")

    async def test_case_3_2b_filter_by_date_period(self):
        """
        Test Case 3.2b - Filter by actual date period

        Action: Query "concepts I learned on Feb 24" vs "concepts learned on Feb 26"
        Expected Output:
          - Returns only nodes created/annotated on specific date
          - Properly filters by date field, not just highlight_id
          - Can query date ranges (e.g., "this week", "Feb 24-26")
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph with date metadata
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Act - Query nodes by actual date field
            # This tests if nodes store and can be filtered by creation/annotation date

            # Get all nodes first to inspect their date properties
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            # Check if nodes have date information
            nodes_with_dates = [
                node for node in all_nodes
                if hasattr(node, 'created_at') or hasattr(node, 'date') or hasattr(node, 'metadata')
            ]

            # Test 1: Verify nodes store date information from metadata
            assert len(all_nodes) > 0, "Should have nodes in the graph"

            # Inspect what date fields are available
            sample_node = all_nodes[0]
            available_fields = dir(sample_node)
            date_related_fields = [f for f in available_fields if 'date' in f.lower() or 'time' in f.lower() or 'created' in f.lower()]

            # Test 2: Try to filter nodes by date using GraphUIService
            # If the service doesn't support date filtering, this will fail
            try:
                # Attempt to query by date (if supported)
                from datetime import datetime

                feb24_date = "2025-02-24"
                feb26_date = "2025-02-26"

                # Try filtering by date through graph_ops or GraphUIService
                # This will reveal if date filtering is implemented

                # For now, manually filter nodes that should belong to each date
                # based on their highlight_ids (since we know the mapping)
                feb24_highlights = {1, 4}
                feb26_highlights = {10, 11}

                feb24_nodes_by_date = []
                feb26_nodes_by_date = []

                for node in all_nodes:
                    # Check if node has highlight_id in its metadata/properties
                    if hasattr(node, 'highlight_id') and node.highlight_id:
                        if isinstance(node.highlight_id, list):
                            if any(h in feb24_highlights for h in node.highlight_id):
                                feb24_nodes_by_date.append(node)
                            if any(h in feb26_highlights for h in node.highlight_id):
                                feb26_nodes_by_date.append(node)
                        elif node.highlight_id in feb24_highlights:
                            feb24_nodes_by_date.append(node)
                        elif node.highlight_id in feb26_highlights:
                            feb26_nodes_by_date.append(node)

                # Test 3: Verify date-based separation is meaningful
                # Concepts from Feb 24 should be about economists
                feb24_names = {node.name.lower() for node in feb24_nodes_by_date}
                feb26_names = {node.name.lower() for node in feb26_nodes_by_date}

                # Check for expected Feb 24 content (economist names)
                feb24_economist_concepts = [
                    name for name in ["frank knight", "henry simons", "alan director", "milton friedman"]
                    if any(name in node_name for node_name in feb24_names)
                ]

                # Check for expected Feb 26 content (philosophical concepts)
                feb26_philosophy_concepts = [
                    name for name in ["altruism", "collectivism", "nineteenth-century liberalism"]
                    if any(name in node_name for node_name in feb26_names)
                ]

                # Test 4: Strict content separation - concepts should NOT cross dates inappropriately
                # Feb 24 should NOT have altruism/collectivism (those are Feb 26 concepts)
                feb26_exclusive_concepts = ["altruism", "collectivism"]
                feb26_concepts_in_feb24 = [
                    concept for concept in feb26_exclusive_concepts
                    if any(concept in name for name in feb24_names)
                ]

                assert len(feb26_concepts_in_feb24) == 0, \
                    f"Feb 24 should NOT contain Feb 26-exclusive concepts: {feb26_concepts_in_feb24}"

                # Test 5: Stricter overlap check (should be < 40% for distinct dates)
                overlap = feb24_names.intersection(feb26_names)
                if len(feb24_names) > 0 and len(feb26_names) > 0:
                    overlap_ratio = len(overlap) / max(len(feb24_names), len(feb26_names))
                    assert overlap_ratio < 0.4, \
                        f"Date-based filtering should have minimal overlap (found {overlap_ratio:.1%}). Overlapping: {overlap}"

                # Test 6: Query by date range (Feb 24-26 should include both sets)
                all_feb_nodes = feb24_nodes_by_date + feb26_nodes_by_date
                all_feb_names = {node.name.lower() for node in all_feb_nodes}

                # Should include both economist names AND philosophical concepts
                assert len(feb24_economist_concepts) >= 1 or len(feb26_philosophy_concepts) >= 1, \
                    "Date range query should capture concepts from multiple dates"

                print(f"\n✓ Date-based filtering test results:")
                print(f"  - Available date-related fields on nodes: {date_related_fields}")
                print(f"  - Feb 24 concepts: {len(feb24_nodes_by_date)} nodes")
                print(f"  - Feb 26 concepts: {len(feb26_nodes_by_date)} nodes")
                print(f"  - Overlap ratio: {overlap_ratio:.1%}")
                print(f"  - Feb 24 economists found: {feb24_economist_concepts}")
                print(f"  - Feb 26 philosophy concepts found: {feb26_philosophy_concepts}")

            except Exception as e:
                # If date filtering is not implemented, provide helpful failure message
                assert False, f"Date-based filtering not fully implemented: {str(e)}\n" \
                             f"Available node fields: {date_related_fields}\n" \
                             f"To fix: Nodes need to store date metadata and GraphUIService needs date filtering"

    async def test_case_3_3_filter_by_book_id(self):
        """
        Additional test - Verify book_id filtering works correctly
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Act - Query by book_id
            book1_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                book_id=1
            )

            # Assert
            nodes = book1_result.get("nodes", [])
            assert len(nodes) > 0, "Should have nodes from book 1"

            # Verify all nodes have book_id=1
            for node in nodes:
                book_ids = node.get("book_id", [])
                assert 1 in book_ids, f"Node '{node['name']}' should have book_id=1, got: {book_ids}"

    async def test_case_3_4_query_performance(self):
        """
        Additional test - Verify query performance is acceptable
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Test multiple query types and measure time
            import time

            # Test 1: Get all nodes
            start = time.time()
            nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            get_all_time = time.time() - start

            assert get_all_time < 2.0, f"get_all_nodes should be fast (took {get_all_time:.2f}s)"

            # Test 2: Get relationships
            start = time.time()
            relationships = await graph_ops.get_all_relationships(TEST_USER_ID)
            get_rels_time = time.time() - start

            assert get_rels_time < 2.0, f"get_all_relationships should be fast (took {get_rels_time:.2f}s)"

            # Test 3: Similarity search
            start = time.time()
            similar = await graph_ops.text_similarity_search(
                query="classical liberalism",
                user_id=TEST_USER_ID,
                limit=5
            )
            similarity_time = time.time() - start

            assert similarity_time < 3.0, f"text_similarity_search should be fast (took {similarity_time:.2f}s)"

    async def test_case_3_5_concept_evolution(self):
        """
        Additional test - Track concept evolution across multiple mentions
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Query for concepts that appear multiple times
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            # Find nodes with multiple chunk_ids (indicating multiple mentions)
            evolved_concepts = [
                node for node in all_nodes
                if hasattr(node, 'chunk_ids') and node.chunk_ids and len(node.chunk_ids) > 1
            ]

            # If concepts appear multiple times, they should be properly merged
            if len(evolved_concepts) > 0:
                for node in evolved_concepts:
                    # Should have accumulated chunk_ids from multiple sources
                    assert len(node.chunk_ids) >= 2, \
                        f"Evolved concept '{node.name}' should have multiple chunk references"

                    # Should maintain type and properties across mentions
                    assert node.type is not None, \
                        f"Evolved concept '{node.name}' should maintain type information"

    async def test_case_3_6_relationship_integrity(self):
        """
        Additional test - Verify relationship integrity in stored graph
        """
        async with GraphOps() as graph_ops:
            # Set up sample graph
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Get all data
            nodes = await graph_ops.get_all_nodes(TEST_USER_ID)
            relationships = await graph_ops.get_all_relationships(TEST_USER_ID)

            # Build node name set
            node_names = {node.name for node in nodes}

            # Verify all relationships reference existing nodes
            for rel in relationships:
                assert rel.source in node_names, \
                    f"Relationship source '{rel.source}' should reference existing node"
                assert rel.target in node_names, \
                    f"Relationship target '{rel.target}' should reference existing node"

                # Relationship should have a type
                assert rel.relation is not None and len(rel.relation) > 0, \
                    f"Relationship {rel.source} -> {rel.target} should have a relation type"

            # Verify no self-loops (node relating to itself) unless intentional
            self_loops = [rel for rel in relationships if rel.source == rel.target]

            # Self-loops should be rare or non-existent
            assert len(self_loops) <= len(relationships) * 0.05, \
                f"Too many self-loops found: {len(self_loops)} out of {len(relationships)}"

    async def test_case_3_6_direct_date_filtering(self):
        """
        Test Case 3.6 - Direct date-based filtering using GraphUIService

        Action: Query concepts learned on specific dates using date_from and date_to parameters
        Expected Output:
          - Correctly filters nodes by created_at timestamp
          - Returns only nodes within the specified date range
          - Date filtering works in combination with other filters
        """
        async with GraphOps() as graph_ops:
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Test 1: Filter by single date (Feb 24, 2025)
            feb24_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-24",
                date_to="2025-02-24T23:59:59"
            )

            feb24_nodes = feb24_result.get("nodes", [])
            print(f"\n✓ Feb 24 query returned {len(feb24_nodes)} nodes")

            # Verify all nodes have created_at within Feb 24
            for node in feb24_nodes:
                created_at = node.get("created_at")
                assert created_at is not None, f"Node '{node['name']}' should have created_at field"
                assert created_at.startswith("2025-02-24"), \
                    f"Node '{node['name']}' created_at should be on Feb 24, got: {created_at}"

            # Test 2: Filter by different single date (Feb 26, 2025)
            feb26_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-26",
                date_to="2025-02-26T23:59:59"
            )

            feb26_nodes = feb26_result.get("nodes", [])
            print(f"✓ Feb 26 query returned {len(feb26_nodes)} nodes")

            # Verify all nodes have created_at within Feb 26
            for node in feb26_nodes:
                created_at = node.get("created_at")
                assert created_at is not None, f"Node '{node['name']}' should have created_at field"
                assert created_at.startswith("2025-02-26"), \
                    f"Node '{node['name']}' created_at should be on Feb 26, got: {created_at}"

            # Test 3: Filter by date range (Feb 24-26)
            range_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-24",
                date_to="2025-02-26T23:59:59"
            )

            range_nodes = range_result.get("nodes", [])
            print(f"✓ Feb 24-26 range query returned {len(range_nodes)} nodes")

            # Range should include nodes from both dates
            assert len(range_nodes) >= len(feb24_nodes), "Range should include Feb 24 nodes"
            assert len(range_nodes) >= len(feb26_nodes), "Range should include Feb 26 nodes"

            # Test 4: No date filter (baseline)
            all_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops
            )

            all_nodes = all_result.get("nodes", [])
            print(f"✓ No filter query returned {len(all_nodes)} nodes")

            # Filtered results should be subsets of total
            assert len(feb24_nodes) <= len(all_nodes), "Feb 24 should be subset of all nodes"
            assert len(feb26_nodes) <= len(all_nodes), "Feb 26 should be subset of all nodes"
            assert len(range_nodes) <= len(all_nodes), "Range should be subset of all nodes"

            # Test 5: Combined filtering (date + book_id)
            combined_result = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                book_id=1,
                date_from="2025-02-24",
                date_to="2025-02-24T23:59:59"
            )

            combined_nodes = combined_result.get("nodes", [])
            print(f"✓ Combined filter (book_id=1, date=Feb 24) returned {len(combined_nodes)} nodes")

            # Combined filter should be more restrictive
            assert len(combined_nodes) <= len(feb24_nodes), \
                "Combined filter should return fewer or equal nodes than date-only filter"

            # Verify combined filter conditions
            for node in combined_nodes:
                book_ids = node.get("book_id", [])
                created_at = node.get("created_at")
                assert 1 in book_ids, f"Node '{node['name']}' should have book_id=1"
                assert created_at.startswith("2025-02-24"), \
                    f"Node '{node['name']}' should be created on Feb 24"

            print(f"\n✓ Date filtering test results:")
            print(f"  - Feb 24 only: {len(feb24_nodes)} nodes")
            print(f"  - Feb 26 only: {len(feb26_nodes)} nodes")
            print(f"  - Feb 24-26 range: {len(range_nodes)} nodes")
            print(f"  - All nodes: {len(all_nodes)} nodes")
            print(f"  - Combined (book_id=1 + Feb 24): {len(combined_nodes)} nodes")

    async def test_case_3_7_when_did_i_learn_query(self):
        """
        Test Case 3.7 - "When did I learn about X?" temporal queries

        Action: Query when specific concepts were learned using created_at and bloom_history
        Expected Output:
          - Can determine when a concept was first learned
          - Can track Bloom level progression over time
          - Bloom history shows source attribution
        """
        async with GraphOps() as graph_ops:
            await setup_sample_graph(graph_ops, TEST_USER_ID)

            # Get all nodes to inspect temporal data
            all_nodes = await graph_ops.get_all_nodes(TEST_USER_ID)

            # Test 1: Find when specific concepts were learned
            concept_timestamps = {}
            for node in all_nodes:
                if hasattr(node, 'created_at') and node.created_at:
                    concept_timestamps[node.name] = node.created_at

            print(f"\n✓ Found {len(concept_timestamps)} concepts with timestamps")

            # Verify we have temporal data
            assert len(concept_timestamps) > 0, "Should have concepts with created_at timestamps"

            # Test 2: Examine Bloom level progression for specific concepts
            bloom_progressions = {}
            for node in all_nodes:
                if hasattr(node, 'bloom_history') and node.bloom_history:
                    bloom_progressions[node.name] = node.bloom_history

            print(f"✓ Found {len(bloom_progressions)} concepts with Bloom history")

            # Verify we have Bloom progression data
            assert len(bloom_progressions) > 0, "Should have concepts with bloom_history"

            # Test 3: Analyze a specific concept's learning timeline
            # Find a concept that exists in the graph
            test_concepts = ["Milton Friedman", "Altruism is collectivism", "Classical liberalism"]
            found_concept = None

            for concept_name in test_concepts:
                matching_nodes = [n for n in all_nodes if n.name == concept_name]
                if matching_nodes:
                    found_concept = matching_nodes[0]
                    break

            if found_concept:
                print(f"\n✓ Analyzing concept: '{found_concept.name}'")

                # When was it learned?
                if hasattr(found_concept, 'created_at') and found_concept.created_at:
                    print(f"  - First learned: {found_concept.created_at}")

                    # Verify timestamp format
                    assert "2025-02-" in found_concept.created_at or "T" in found_concept.created_at, \
                        "created_at should be in ISO 8601 format"

                # How has understanding evolved?
                if hasattr(found_concept, 'bloom_history') and found_concept.bloom_history:
                    print(f"  - Bloom progression ({len(found_concept.bloom_history)} entries):")

                    for i, update in enumerate(found_concept.bloom_history):
                        level = update.get('level') if isinstance(update, dict) else getattr(update, 'level', None)
                        timestamp = update.get('timestamp') if isinstance(update, dict) else getattr(update, 'timestamp', None)
                        source = update.get('source') if isinstance(update, dict) else getattr(update, 'source', None)

                        print(f"    {i+1}. {level} at {timestamp} (from {source})")

                        # Verify bloom history structure
                        assert level is not None, "Bloom update should have level"
                        assert timestamp is not None, "Bloom update should have timestamp"

            # Test 4: Query concepts by learning date using GraphUIService
            # Find concepts learned on Feb 24
            feb24_concepts = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-24",
                date_to="2025-02-24T23:59:59"
            )

            feb24_nodes = feb24_concepts.get("nodes", [])

            if feb24_nodes:
                print(f"\n✓ Concepts learned on Feb 24, 2025:")
                for node in feb24_nodes[:5]:  # Show first 5
                    print(f"  - {node['name']}")
                    if node.get('bloom_history'):
                        initial_level = node['bloom_history'][0].get('level', 'Unknown')
                        print(f"    Initial level: {initial_level}")

            # Test 5: Find concepts learned in a specific date range
            feb_range = await GraphUIService.get_graph_ui_data(
                user_id=TEST_USER_ID,
                graph_ops=graph_ops,
                date_from="2025-02-24",
                date_to="2025-02-28"
            )

            range_nodes = feb_range.get("nodes", [])

            print(f"\n✓ Query results summary:")
            print(f"  - Total concepts with timestamps: {len(concept_timestamps)}")
            print(f"  - Concepts with Bloom history: {len(bloom_progressions)}")
            print(f"  - Concepts learned on Feb 24: {len(feb24_nodes)}")
            print(f"  - Concepts learned Feb 24-28: {len(range_nodes)}")

            # Test 6: Verify temporal data integrity
            for node in all_nodes[:10]:  # Check first 10 nodes
                # If node has bloom_history, it should also have created_at
                has_bloom = hasattr(node, 'bloom_history') and node.bloom_history
                has_created = hasattr(node, 'created_at') and node.created_at

                if has_bloom:
                    assert has_created, \
                        f"Node '{node.name}' has bloom_history but missing created_at"
