"""
Test script to find and clean up empty/placeholder UserNote nodes.

This script:
1. Finds UserNote nodes with empty or placeholder names
2. Shows examples
3. Provides a cleanup query
"""
import asyncio
from persona.core.neo4j_database import Neo4jConnectionManager

async def check_empty_usernotes():
    # Initialize Neo4j manager
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()

    # Use the test user ID
    user_id = "18"

    print(f"Checking for empty/placeholder UserNote nodes for user {user_id}...")
    print("=" * 80)

    async with neo4j_manager._ensure_driver().session() as session:
        # Find empty or placeholder UserNotes
        query = """
        MATCH (n:UserNote)
        WHERE n.UserId = $user_id
          AND (
            trim(n.name) = ''
            OR toLower(trim(n.name)) IN ['no user note provided', 'no note', 'none', 'n/a']
          )
        RETURN n.name as name, id(n) as node_id, count(*) as count
        """

        result = await session.run(query, user_id=user_id)
        records = await result.values()

        if records and records[0][2] > 0:
            print(f"\n⚠ Found {records[0][2]} empty/placeholder UserNote nodes:")
            print("-" * 80)
            for record in records:
                name = repr(record[0])  # Use repr to show empty strings clearly
                node_id = record[1]
                print(f"  • Name: {name} (node_id: {node_id})")

            print("\n" + "=" * 80)
            print("Cleaning up these nodes...")

            # Delete empty/placeholder UserNotes
            delete_query = """
            MATCH (n:UserNote)
            WHERE n.UserId = $user_id
              AND (
                trim(n.name) = ''
                OR toLower(trim(n.name)) IN ['no user note provided', 'no note', 'none', 'n/a']
              )
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """

            result = await session.run(delete_query, user_id=user_id)
            record = await result.single()
            deleted = record['deleted_count'] if record else 0

            print(f"✓ Deleted {deleted} empty/placeholder UserNote nodes")
        else:
            print("\n✓ No empty/placeholder UserNote nodes found!")

        # Show remaining UserNotes
        print("\n" + "=" * 80)
        print("Remaining UserNote nodes:")
        print("-" * 80)

        remaining_query = """
        MATCH (n:UserNote)
        WHERE n.UserId = $user_id
        RETURN n.name as name, n.discipline as discipline
        ORDER BY n.name
        LIMIT 10
        """

        result = await session.run(remaining_query, user_id=user_id)
        records = await result.values()

        if records:
            for record in records:
                name = record[0][:70] + "..." if len(record[0]) > 70 else record[0]
                discipline = record[1] if record[1] else "N/A"
                print(f"  • {name} [{discipline}]")
        else:
            print("  (No UserNote nodes in database)")

    await neo4j_manager.close()
    print("\n" + "=" * 80)
    print("Check complete!")

if __name__ == "__main__":
    asyncio.run(check_empty_usernotes())
