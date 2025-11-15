"""
Test script to verify Term extraction improvements.

This script:
1. Counts existing Term nodes in the graph
2. Shows examples of extracted Terms
3. Provides statistics on Term distribution by discipline
"""
import asyncio
from persona.core.neo4j_database import Neo4jConnectionManager

async def test_term_extraction():
    # Initialize Neo4j manager
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()

    # Use the test user ID
    user_id = "18"

    print(f"Analyzing Term extraction for user {user_id}...")
    print("=" * 80)

    async with neo4j_manager._ensure_driver().session() as session:
        # Count total Terms
        count_query = """
        MATCH (n:Term)
        WHERE n.UserId = $user_id
        RETURN count(n) as term_count
        """
        result = await session.run(count_query, user_id=user_id)
        record = await result.single()
        term_count = record['term_count'] if record else 0

        print(f"\nTotal Term nodes: {term_count}")

        # Get examples of Terms
        examples_query = """
        MATCH (n:Term)
        WHERE n.UserId = $user_id
        RETURN n.name as name, n.discipline as discipline, n.confidence as confidence
        ORDER BY n.confidence DESC
        LIMIT 20
        """
        result = await session.run(examples_query, user_id=user_id)
        records = await result.values()

        if records:
            print("\nExample Terms (top 20 by confidence):")
            print("-" * 80)
            for record in records:
                name = record[0][:60] + "..." if len(record[0]) > 60 else record[0]
                discipline = record[1] if record[1] else "N/A"
                confidence = record[2] if record[2] else 0
                print(f"  • {name:<65} [{discipline}] ({confidence:.2f})")

        # Get Terms by discipline
        discipline_query = """
        MATCH (n:Term)
        WHERE n.UserId = $user_id
        RETURN n.discipline as discipline, count(n) as count
        ORDER BY count DESC
        """
        result = await session.run(discipline_query, user_id=user_id)
        records = await result.values()

        if records:
            print("\n" + "=" * 80)
            print("Terms by Discipline:")
            print("-" * 80)
            for record in records:
                discipline = record[0] if record[0] else "Not specified"
                count = record[1]
                print(f"  {discipline:<30} {count:>3} terms")

        # Check for specific Term types the user mentioned
        print("\n" + "=" * 80)
        print("Checking for specific Term types mentioned by user:")
        print("-" * 80)

        specific_terms = [
            "Libertarianism", "Communism", "Economics", "Psychology",
            "Philosophy", "Sociology", "Market", "Louvre", "Bulgaria"
        ]

        for term_name in specific_terms:
            check_query = """
            MATCH (n:Term)
            WHERE n.UserId = $user_id AND toLower(n.name) CONTAINS toLower($term_name)
            RETURN n.name as name, n.discipline as discipline
            """
            result = await session.run(check_query, user_id=user_id, term_name=term_name)
            record = await result.single()

            if record:
                print(f"  ✓ Found: {record['name']} [{record['discipline']}]")
            else:
                print(f"  ✗ Not found: {term_name}")

    await neo4j_manager.close()
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("\nNote: To test improved extraction, trigger a reingestion with new content")
    print("      The updated prompts should extract 3-10 Terms per document.")

if __name__ == "__main__":
    asyncio.run(test_term_extraction())
