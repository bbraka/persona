"""
Test script to verify transitive RELATED_TO cleanup functionality.

This script:
1. Counts existing transitive RELATED_TO edges (where alternative 2-3 hop paths exist)
2. Runs the enhanced cleanup method
3. Verifies that transitive RELATED_TO edges were removed
"""
import asyncio
from persona.core.neo4j_database import Neo4jConnectionManager

async def test_transitive_cleanup():
    # Initialize Neo4j manager
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()

    # Use the test user ID
    user_id = "18"

    print(f"Testing transitive RELATED_TO cleanup for user {user_id}...")
    print("=" * 80)

    # Query to count transitive RELATED_TO relationships BEFORE cleanup
    count_query = """
    MATCH (a)-[r:RELATED_TO]->(b)
    WHERE a.UserId = $user_id
      AND EXISTS {
        MATCH path = (a)-[*2..3]->(b)
        WHERE NONE(rel IN relationships(path) WHERE type(rel) = 'RELATED_TO')
      }
    RETURN count(r) as transitive_count,
           collect(DISTINCT {source: a.name, target: b.name})[..5] as examples
    """

    async with neo4j_manager._ensure_driver().session() as session:
        print("\nStep 1: Counting transitive RELATED_TO edges...")
        result = await session.run(count_query, user_id=user_id)
        record = await result.single()

        if record:
            transitive_count = record['transitive_count']
            examples = record['examples']

            print(f"\nFound {transitive_count} transitive RELATED_TO relationships")

            if transitive_count > 0:
                print("\nExamples of transitive edges (nodes connected via 2-3 hop paths):")
                for ex in examples:
                    source_name = ex['source'][:60] + "..." if len(ex['source']) > 60 else ex['source']
                    target_name = ex['target'][:60] + "..." if len(ex['target']) > 60 else ex['target']
                    print(f"  • {source_name}")
                    print(f"    --[RELATED_TO]--> {target_name}")
                    print(f"    (already connected via alternative path)")

            # Run the enhanced cleanup
            print(f"\n" + "=" * 80)
            print("Step 2: Running enhanced cleanup...")
            print("=" * 80)
            deleted_count = await neo4j_manager.delete_redundant_related_to_relationships(user_id)

            print(f"\n" + "=" * 80)
            print(f"✓ Cleanup complete: {deleted_count} total redundant RELATED_TO relationships removed")
            print("=" * 80)

            # Verify no transitive RELATED_TO remain
            print("\nStep 3: Verifying transitive edges were removed...")
            result = await session.run(count_query, user_id=user_id)
            record = await result.single()
            remaining = record['transitive_count'] if record else 0

            if remaining == 0:
                print(f"✓ Verification passed: No transitive RELATED_TO relationships remain")
            else:
                print(f"⚠ Warning: {remaining} transitive RELATED_TO relationships still exist")
                print("   (These may be new edges or the query needs adjustment)")
        else:
            print("No data found for user")

    await neo4j_manager.close()
    print("\n" + "=" * 80)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_transitive_cleanup())
