"""
Quick speed test for Layer 1 deduplication.
Tests how fast it processes nodes with and without deduplication.
"""
import asyncio
import time
from persona.core.graph_ops import GraphOps
from persona.models.schema import Node

async def test_dedup_speed():
    """Test the speed of Layer 1 deduplication."""

    # Test user
    user_id = "speed_test_user"

    # Create similar test nodes
    test_nodes = [
        Node(name="Joy of reunion", type="Emotion", description="Feeling happy when reuniting"),
        Node(name="Joy of reuniting", type="Emotion", description="Happy feeling of reunion"),
        Node(name="Happiness of reunion", type="Emotion", description="Joy when reuniting with someone"),
        Node(name="Joy in unexpected reunions", type="Emotion", description="Surprise joy of reunion"),
        Node(name="Father-son reunion brings joy", type="Emotion", description="Joy from family reunion"),
        Node(name="Completely different concept", type="Concept", description="Not related at all"),
        Node(name="Another different thing", type="Concept", description="Also unrelated"),
    ]

    async with GraphOps() as graph_ops:
        print(f"\n{'='*60}")
        print("Testing Layer 1 Deduplication Speed")
        print(f"{'='*60}\n")

        # Clean up any existing test data
        print("Cleaning up test data...")
        await graph_ops.neo4j_manager.execute_query(
            "MATCH (n) WHERE n.user_id = $user_id DETACH DELETE n",
            {"user_id": user_id}
        )

        # Test adding nodes with deduplication
        print(f"\nAdding {len(test_nodes)} nodes with Layer 1 deduplication enabled...\n")

        start_time = time.time()

        for i, node in enumerate(test_nodes, 1):
            node_start = time.time()

            result = await graph_ops.add_nodes([node], user_id=user_id)

            node_time = time.time() - node_start

            status = "✓ Created" if result["nodes_created"] > 0 else "⊘ Skipped (duplicate)"
            print(f"  [{i}/{len(test_nodes)}] {node.name[:40]:<40} | {node_time:.3f}s | {status}")

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per node: {total_time/len(test_nodes):.3f}s")
        print(f"{'='*60}\n")

        # Check how many nodes were actually created
        result = await graph_ops.neo4j_manager.execute_query(
            "MATCH (n) WHERE n.user_id = $user_id RETURN count(n) as count",
            {"user_id": user_id}
        )
        nodes_created = result[0]["count"]
        nodes_skipped = len(test_nodes) - nodes_created

        print(f"Nodes created: {nodes_created}")
        print(f"Nodes skipped (as duplicates): {nodes_skipped}")
        print(f"Deduplication efficiency: {(nodes_skipped/len(test_nodes))*100:.1f}% reduction\n")

        # Clean up
        print("Cleaning up test data...")
        await graph_ops.neo4j_manager.execute_query(
            "MATCH (n) WHERE n.user_id = $user_id DETACH DELETE n",
            {"user_id": user_id}
        )

        print("✓ Speed test complete!\n")

if __name__ == "__main__":
    asyncio.run(test_dedup_speed())
