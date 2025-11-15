"""
Test script to verify redundant RELATED_TO cleanup functionality.

This script:
1. Counts existing parallel edges (RELATED_TO + other relationships between same nodes)
2. Runs the cleanup method
3. Verifies that redundant RELATED_TO edges were removed
"""
import asyncio
import os
from persona.core.neo4j_database import Neo4jConnectionManager
from server.config import config

async def test_cleanup():
    # Initialize Neo4j manager
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()

    # Use the test user ID (adjust if needed)
    user_id = "18"

    print(f"Testing redundant RELATED_TO cleanup for user {user_id}...")
    print("=" * 60)

    # Query to count redundant RELATED_TO relationships BEFORE cleanup
    count_query = """
    MATCH (a)-[r1:RELATED_TO]->(b)
    WHERE a.UserId = $user_id
    WITH a, b, r1
    MATCH (a)-[r2]->(b)
    WHERE type(r2) <> 'RELATED_TO'
    RETURN count(r1) as redundant_count,
           collect(DISTINCT {source: a.name, target: b.name, other_rel: type(r2)}) as examples
    """

    async with neo4j_manager._ensure_driver().session() as session:
        result = await session.run(count_query, user_id=user_id)
        record = await result.single()

        if record:
            redundant_count = record['redundant_count']
            examples = record['examples'][:5]  # Show first 5 examples

            print(f"\nFound {redundant_count} redundant RELATED_TO relationships")

            if redundant_count > 0:
                print("\nExamples of parallel edges:")
                for ex in examples:
                    print(f"  • {ex['source']} --[RELATED_TO]--> {ex['target']}")
                    print(f"    (also has {ex['other_rel']} relationship)")

            # Run the cleanup
            print(f"\nRunning cleanup...")
            deleted_count = await neo4j_manager.delete_redundant_related_to_relationships(user_id)

            print(f"\n✓ Cleanup complete: {deleted_count} redundant RELATED_TO relationships removed")

            # Verify no redundant RELATED_TO remain
            result = await session.run(count_query, user_id=user_id)
            record = await result.single()
            remaining = record['redundant_count'] if record else 0

            if remaining == 0:
                print(f"✓ Verification passed: No redundant RELATED_TO relationships remain")
            else:
                print(f"⚠ Warning: {remaining} redundant RELATED_TO relationships still exist")
        else:
            print("No data found for user")

    await neo4j_manager.close()
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_cleanup())
