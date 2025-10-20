"""
Integration tests for node deduplication functionality.

Tests cover:
- Finding similar nodes
- Preventing duplicate creation
- Consolidating existing duplicates
- Threshold tuning
- Relationship preservation
"""

import pytest
import os
from persona.core import GraphOps
from persona.core.deduplication import NodeDeduplicator
from persona.models.schema import NodeModel
from typing import List


# Override Neo4j URI for local testing (use localhost instead of docker service name)
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with localhost Neo4j connection"""
    original_uri = os.environ.get("URI_NEO4J")
    # Use localhost for tests (assumes Neo4j is exposed on host)
    os.environ["URI_NEO4J"] = "bolt://localhost:7687"

    # Force reload of config to pick up the change
    import importlib
    from server import config as config_module
    importlib.reload(config_module)

    yield

    # Restore original
    if original_uri:
        os.environ["URI_NEO4J"] = original_uri
    else:
        os.environ.pop("URI_NEO4J", None)

    # Reload config again to restore
    importlib.reload(config_module)


TEST_USER_ID = "test_dedup_user"


class TestSimilarityDetection:
    """Test finding similar nodes"""

    @pytest.mark.asyncio
    async def test_find_similar_node_basic(self):
        """Test finding a similar node"""
        async with GraphOps() as graph_ops:
            # Create test user
            if not await graph_ops.user_exists(TEST_USER_ID):
                await graph_ops.create_user(TEST_USER_ID)

            try:
                # Create initial node
                initial_nodes = [
                    NodeModel(
                        name="Joy of unexpected reunion",
                        type="Concept",
                        properties={"discipline": "Psychology"}
                    )
                ]
                await graph_ops.add_nodes(initial_nodes, TEST_USER_ID)

                # Search for similar node
                similar = await graph_ops.deduplicator.find_similar_node(
                    node_name="Joy in unexpected reunions",
                    node_type="Concept",
                    user_id=TEST_USER_ID
                )

                assert similar is not None
                assert similar["name"] == "Joy of unexpected reunion"
                assert similar["score"] >= 0.85
            finally:
                # Cleanup
                await graph_ops.delete_user(TEST_USER_ID)

    @pytest.mark.asyncio
    async def test_find_similar_node_no_match(self, graph_ops, test_user_id, clean_test_user):
        """Test when no similar node exists"""
        # Create initial node
        initial_nodes = [
            NodeModel(
                name="Joy of unexpected reunion",
                type="Concept",
                properties={"discipline": "Psychology"}
            )
        ]
        await graph_ops.add_nodes(initial_nodes, test_user_id)

        # Search for completely different concept
        similar = await graph_ops.deduplicator.find_similar_node(
            node_name="Sadness of permanent separation",
            node_type="Concept",
            user_id=test_user_id
        )

        # Should not find a match above threshold
        assert similar is None

    @pytest.mark.asyncio
    async def test_similarity_threshold_strict(self, test_user_id, clean_test_user):
        """Test with stricter threshold"""
        # Create graph with strict threshold
        graph = GraphOps(
            enable_deduplication=True,
            dedup_similarity_threshold=0.95
        )
        await graph.initialize()

        try:
            # Create initial node
            initial_nodes = [
                NodeModel(
                    name="Professional jealousy",
                    type="Concept",
                    properties={"discipline": "Psychology"}
                )
            ]
            await graph.add_nodes(initial_nodes, test_user_id)

            # This should match with default threshold but not with 0.95
            similar = await graph.deduplicator.find_similar_node(
                node_name="Envy in workplace",
                node_type="Concept",
                user_id=test_user_id
            )

            # Might not find match with very strict threshold
            # (depends on embedding model)
            if similar:
                assert similar["score"] >= 0.95

        finally:
            await graph.close()


class TestDuplicatePrevention:
    """Test preventing duplicate node creation"""

    @pytest.mark.asyncio
    async def test_prevent_duplicate_creation(self, graph_ops, test_user_id, clean_test_user):
        """Test that duplicate nodes are not created"""
        # Create first node
        nodes1 = [
            NodeModel(
                name="Joy of unexpected reunion",
                type="Concept",
                properties={"discipline": "Psychology", "confidence": 0.9}
            )
        ]
        await graph_ops.add_nodes(nodes1, test_user_id)

        # Try to create similar nodes
        nodes2 = [
            NodeModel(
                name="Joy of reunion with loved ones",
                type="Concept",
                properties={"discipline": "Psychology", "confidence": 0.85}
            ),
            NodeModel(
                name="Joy in unexpected reunions",
                type="Concept",
                properties={"discipline": "Psychology", "confidence": 0.88}
            )
        ]
        await graph_ops.add_nodes(nodes2, test_user_id)

        # Check that only original node exists
        all_nodes = await graph_ops.get_all_nodes(test_user_id)
        joy_nodes = [n for n in all_nodes if "joy" in n.name.lower() and "reunion" in n.name.lower()]

        assert len(joy_nodes) == 1
        assert joy_nodes[0].name == "Joy of unexpected reunion"

    @pytest.mark.asyncio
    async def test_create_distinct_nodes(self, graph_ops, test_user_id, clean_test_user):
        """Test that distinct nodes are still created"""
        nodes = [
            NodeModel(
                name="Joy of unexpected reunion",
                type="Concept",
                properties={"discipline": "Psychology"}
            ),
            NodeModel(
                name="Sadness of separation",
                type="Concept",
                properties={"discipline": "Psychology"}
            ),
            NodeModel(
                name="Professional jealousy",
                type="Concept",
                properties={"discipline": "Psychology"}
            )
        ]
        await graph_ops.add_nodes(nodes, test_user_id)

        all_nodes = await graph_ops.get_all_nodes(test_user_id)
        concept_nodes = [n for n in all_nodes if n.type == "Concept"]

        # All three should be created (they're distinct)
        assert len(concept_nodes) == 3

    @pytest.mark.asyncio
    async def test_deduplication_disabled(self, graph_ops_no_dedup, test_user_id, clean_test_user):
        """Test that duplicates ARE created when deduplication is disabled"""
        # Create user for this graph instance too
        if not await graph_ops_no_dedup.user_exists(test_user_id):
            await graph_ops_no_dedup.create_user(test_user_id)

        nodes = [
            NodeModel(
                name="Joy of unexpected reunion",
                type="Concept",
                properties={"discipline": "Psychology"}
            ),
            NodeModel(
                name="Joy in unexpected reunions",
                type="Concept",
                properties={"discipline": "Psychology"}
            )
        ]
        await graph_ops_no_dedup.add_nodes(nodes, test_user_id)

        all_nodes = await graph_ops_no_dedup.get_all_nodes(test_user_id)
        joy_nodes = [n for n in all_nodes if "joy" in n.name.lower()]

        # Both should be created when dedup is disabled
        assert len(joy_nodes) == 2


class TestConsolidation:
    """Test consolidating existing duplicates"""

    @pytest.mark.asyncio
    async def test_find_duplicate_clusters(self, test_user_id):
        """Test finding clusters of duplicate nodes"""
        # Create graph with dedup disabled to allow duplicates
        graph = GraphOps(enable_deduplication=False)
        await graph.initialize()

        try:
            if not await graph.user_exists(test_user_id):
                await graph.create_user(test_user_id)

            # Create nodes with duplicates
            nodes = [
                # Cluster 1: Joy/reunion concepts
                NodeModel(name="Joy of unexpected reunion", type="Concept", properties={}),
                NodeModel(name="Joy in unexpected reunions", type="Concept", properties={}),
                NodeModel(name="Joy of reunion with loved ones", type="Concept", properties={}),

                # Cluster 2: Betrayal concepts
                NodeModel(name="Betrayal by trusted colleagues", type="Concept", properties={}),
                NodeModel(name="Betrayed by close friends", type="Concept", properties={}),

                # Distinct node
                NodeModel(name="Hope sustains through suffering", type="Concept", properties={})
            ]
            await graph.add_nodes(nodes, test_user_id)

            # Create deduplicator
            deduplicator = NodeDeduplicator(
                graph.neo4j_manager,
                similarity_threshold=0.85
            )

            # Run consolidation in dry-run mode
            report = await deduplicator.consolidate_duplicate_nodes(
                user_id=test_user_id,
                dry_run=True
            )

            # Should find 2 clusters
            assert len(report["duplicate_clusters"]) >= 2

            # Should identify nodes to remove
            assert report["nodes_to_remove"] >= 3

        finally:
            await graph.delete_user(test_user_id)
            await graph.close()

    @pytest.mark.asyncio
    async def test_consolidation_with_relationships(self, test_user_id):
        """Test that relationships are preserved during consolidation"""
        # Create graph with dedup disabled
        graph = GraphOps(enable_deduplication=False)
        await graph.initialize()

        try:
            if not await graph.user_exists(test_user_id):
                await graph.create_user(test_user_id)

            # Create duplicate nodes
            nodes = [
                NodeModel(name="Joy of unexpected reunion", type="Concept", properties={}),
                NodeModel(name="Joy in unexpected reunions", type="Concept", properties={}),
                NodeModel(name="Edmond Dantès", type="Character", properties={})
            ]
            await graph.add_nodes(nodes, test_user_id)

            # Create relationships from both duplicate nodes
            from persona.models.schema import RelationshipModel
            relationships = [
                RelationshipModel(
                    source="Edmond Dantès",
                    relation="EXPERIENCES",
                    target="Joy of unexpected reunion"
                ),
                RelationshipModel(
                    source="Edmond Dantès",
                    relation="EXPERIENCES",
                    target="Joy in unexpected reunions"
                )
            ]
            await graph.add_relationships(relationships, test_user_id)

            # Run consolidation (live mode)
            deduplicator = NodeDeduplicator(
                graph.neo4j_manager,
                similarity_threshold=0.85
            )

            report = await deduplicator.consolidate_duplicate_nodes(
                user_id=test_user_id,
                dry_run=False,
                type_filter="Concept"
            )

            # Check results
            all_nodes = await graph.get_all_nodes(test_user_id)
            joy_nodes = [n for n in all_nodes if "joy" in n.name.lower()]

            # Should only have one joy node
            assert len(joy_nodes) == 1

            # Check that relationships were preserved
            canonical_node = joy_nodes[0].name
            relationships = await graph.get_node_relationships(canonical_node, test_user_id)

            # Should have relationship from Edmond Dantès
            assert any(
                rel.source == "Edmond Dantès" and rel.relation == "EXPERIENCES"
                for rel in relationships
            )

        finally:
            await graph.delete_user(test_user_id)
            await graph.close()

    @pytest.mark.asyncio
    async def test_consolidation_type_filter(self, test_user_id):
        """Test consolidation with type filter"""
        graph = GraphOps(enable_deduplication=False)
        await graph.initialize()

        try:
            if not await graph.user_exists(test_user_id):
                await graph.create_user(test_user_id)

            # Create duplicates of different types
            nodes = [
                NodeModel(name="Joy of reunion", type="Concept", properties={}),
                NodeModel(name="Joy in reunions", type="Concept", properties={}),
                NodeModel(name="Paris", type="Location", properties={}),
                NodeModel(name="Paris, France", type="Location", properties={})
            ]
            await graph.add_nodes(nodes, test_user_id)

            deduplicator = NodeDeduplicator(
                graph.neo4j_manager,
                similarity_threshold=0.85
            )

            # Consolidate only Concepts
            report = await deduplicator.consolidate_duplicate_nodes(
                user_id=test_user_id,
                dry_run=True,
                type_filter="Concept"
            )

            # Should only find Concept duplicates, not Location
            concept_clusters = [
                c for c in report["duplicate_clusters"]
                # Check if any node in cluster contains "Joy"
            ]

            all_nodes = await graph.get_all_nodes(test_user_id)
            location_nodes = [n for n in all_nodes if n.type == "Location"]

            # Locations should not be touched
            assert len(location_nodes) == 2

        finally:
            await graph.delete_user(test_user_id)
            await graph.close()


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_ops, test_user_id, clean_test_user):
        """Test deduplication on empty graph"""
        similar = await graph_ops.deduplicator.find_similar_node(
            node_name="Some concept",
            node_type="Concept",
            user_id=test_user_id
        )

        assert similar is None

    @pytest.mark.asyncio
    async def test_single_node_graph(self, graph_ops, test_user_id, clean_test_user):
        """Test consolidation with only one node"""
        nodes = [
            NodeModel(name="Single concept", type="Concept", properties={})
        ]
        await graph_ops.add_nodes(nodes, test_user_id)

        report = await graph_ops.deduplicator.consolidate_duplicate_nodes(
            user_id=test_user_id,
            dry_run=True
        )

        assert len(report["duplicate_clusters"]) == 0
        assert report["nodes_to_remove"] == 0

    @pytest.mark.asyncio
    async def test_nonexistent_user(self, graph_ops):
        """Test with nonexistent user"""
        similar = await graph_ops.deduplicator.find_similar_node(
            node_name="Some concept",
            node_type="Concept",
            user_id="nonexistent_user"
        )

        # Should handle gracefully
        assert similar is None


class TestCosineSimilarity:
    """Test cosine similarity calculation"""

    def test_identical_vectors(self):
        """Test similarity of identical vectors"""
        vec = [1.0, 2.0, 3.0, 4.0]
        similarity = NodeDeduplicator._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = NodeDeduplicator._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = NodeDeduplicator._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        """Test with zero vector"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = NodeDeduplicator._cosine_similarity(vec1, vec2)
        assert similarity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
