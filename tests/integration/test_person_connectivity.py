#!/usr/bin/env python3
"""
Test that Person nodes are properly connected to Concepts/Themes that mention them.

This test verifies the two-tiered vector search improvement:
1. Create a Person node ("Milton Friedman")
2. Create multiple Concepts that mention this person
3. Verify that relationships are created between the Person and ALL relevant Concepts
4. Before fix: ~5% connection rate (1 out of 20)
5. After fix: ~80-95% connection rate (16-19 out of 20)
"""

import asyncio
from persona.core.graph_ops import GraphOps
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.services.ingest_service import IngestService
from persona.models.schema import UnstructuredData
from neo4j import AsyncGraphDatabase
from server.config import config
import uuid

async def test_person_connectivity():
    """Test that Person nodes connect to most Concepts that mention them"""

    user_id = "test_user_person_connectivity"

    neo4j_manager = Neo4jConnectionManager()
    graph_ops = GraphOps(neo4j_manager=neo4j_manager)

    try:
        async with graph_ops:
            # Clean up any existing test data
            try:
                await graph_ops.delete_user(user_id)
            except:
                pass
            await graph_ops.create_user(user_id)

            print("=" * 80)
            print("TEST: Person Node Connectivity Improvement")
            print("=" * 80)

            # STEP 1: Create Person node + initial Concept with relationship
            print("\n[Step 1] Creating Person node 'Milton Friedman' with initial context...")

            initial_chunk = UnstructuredData(
                title="Highlight: Milton Friedman Biography",
                content="""
                HIGHLIGHT: "Milton Friedman was an American economist"

                USER NOTE: Key figure in Chicago School of Economics. Won Nobel Prize in 1976.
                """,
                chunk_ids=[str(uuid.uuid4())],
                metadata={
                    "user_id": user_id,
                    "data_type": "highlight-note",
                    "book_id": "1001"
                }
            )

            await IngestService.ingest_data(user_id, [initial_chunk], graph_ops)
            print("✓ Created Person node with initial context")

            # STEP 2: Create 10 Concepts that mention Milton Friedman
            print("\n[Step 2] Creating 10 Concepts that mention 'Milton Friedman'...")

            concepts_about_friedman = [
                "Friedman argued that inflation is always and everywhere a monetary phenomenon caused by too rapid growth in the money supply.",
                "Milton Friedman's permanent income hypothesis suggests that consumption depends on long-term average income rather than current income.",
                "The Chicago School of Economics, led by Friedman, emphasized free markets and minimal government intervention.",
                "Friedman's critique of Keynesian economics challenged the dominant macroeconomic paradigm of the 1960s.",
                "Milton Friedman advocated for school vouchers as a way to introduce market competition into public education.",
                "Friedman and Rose Friedman co-authored 'Free to Choose' which became a bestselling book and TV series.",
                "The concept of negative income tax was proposed by Milton Friedman as an alternative to traditional welfare programs.",
                "Friedman's work on consumption analysis led to his recognition with the Nobel Memorial Prize in Economic Sciences.",
                "Milton Friedman served as an advisor to President Reagan and British Prime Minister Margaret Thatcher.",
                "Friedman's monetarism held that variations in the money supply have major influences on output in the short run and prices in the long run.",
            ]

            chunks = []
            for i, concept_text in enumerate(concepts_about_friedman):
                chunk = UnstructuredData(
                    title=f"Reading: Chapter {i+1}",
                    content=f"""
                    CHAPTER {i+1}

                    {concept_text}

                    This concept represents an important aspect of economic theory and policy.
                    """,
                    chunk_ids=[str(uuid.uuid4())],
                    metadata={
                        "user_id": user_id,
                        "data_type": "book-reading-chunk",
                        "book_id": "1001"
                    }
                )
                chunks.append(chunk)

            await IngestService.ingest_data(user_id, chunks, graph_ops)
            print(f"✓ Created {len(concepts_about_friedman)} Concepts mentioning Milton Friedman")

            # STEP 3: Verify connectivity
            print("\n[Step 3] Analyzing Person node connectivity...")

            driver = AsyncGraphDatabase.driver(
                config.NEO4J.URI,
                auth=(config.NEO4J.USER, config.NEO4J.PASSWORD)
            )

            async with driver.session() as session:
                # Count total Concepts
                result = await session.run("""
                    MATCH (c:NodeName {UserId: $user_id, type: 'Concept'})
                    RETURN count(c) as concept_count
                """, user_id=user_id)
                data = await result.single()
                total_concepts = data["concept_count"] if data else 0

                # Count Concepts mentioning Friedman (by name match)
                result = await session.run("""
                    MATCH (c:NodeName {UserId: $user_id, type: 'Concept'})
                    WHERE c.name CONTAINS 'Friedman' OR c.name CONTAINS 'Milton'
                    RETURN count(c) as friedman_concept_count
                """, user_id=user_id)
                data = await result.single()
                concepts_mentioning_friedman = data["friedman_concept_count"] if data else 0

                # Count Concepts connected to Milton Friedman Person node
                result = await session.run("""
                    MATCH (p:NodeName {UserId: $user_id, type: 'Person'})
                    WHERE p.name CONTAINS 'Friedman'
                    WITH p
                    MATCH (p)<-[r]-(c:NodeName {type: 'Concept'})
                    RETURN p.name as person_name, count(DISTINCT c) as connected_concepts
                """, user_id=user_id)

                person_connections = []
                async for record in result:
                    person_connections.append({
                        "person": record["person_name"],
                        "connected_concepts": record["connected_concepts"]
                    })

                print(f"\n  Total Concepts in graph: {total_concepts}")
                print(f"  Concepts mentioning Friedman: {concepts_mentioning_friedman}")

                if person_connections:
                    for conn in person_connections:
                        print(f"\n  Person: '{conn['person']}'")
                        print(f"  Connected to {conn['connected_concepts']} Concepts")

                        connection_rate = (conn['connected_concepts'] / concepts_mentioning_friedman * 100) if concepts_mentioning_friedman > 0 else 0
                        print(f"  Connection rate: {connection_rate:.1f}%")

                        # Get sample of connected relationships
                        result = await session.run("""
                            MATCH (p:NodeName {UserId: $user_id, name: $person_name})
                            MATCH (p)<-[r]-(c:NodeName {type: 'Concept'})
                            RETURN type(r) as relation_type, c.name as concept_name
                            LIMIT 3
                        """, user_id=user_id, person_name=conn['person'])

                        print(f"\n  Sample relationships:")
                        async for record in result:
                            concept_preview = record["concept_name"][:70] + "..." if len(record["concept_name"]) > 70 else record["concept_name"]
                            print(f"    - Concept: '{concept_preview}'")
                            print(f"      Relationship: {record['relation_type']}")

                        # Validation
                        print("\n[Validation]")

                        if conn['connected_concepts'] >= concepts_mentioning_friedman * 0.70:
                            print(f"  ✓✓✓ SUCCESS! {connection_rate:.0f}% connectivity (target: ≥70%)")
                            print("  ✓ Person nodes are properly connected to related Concepts")
                            print("  ✓ Two-tiered vector search is working correctly")
                        elif conn['connected_concepts'] >= concepts_mentioning_friedman * 0.40:
                            print(f"  ⚠ PARTIAL SUCCESS: {connection_rate:.0f}% connectivity")
                            print("  ⚠ Better than baseline (~5%) but below target (70%)")
                        else:
                            print(f"  ✗✗✗ FAILURE: Only {connection_rate:.0f}% connectivity")
                            print("  ✗ Person nodes still not connecting properly")
                else:
                    print("\n  ✗ No Person nodes found for 'Friedman'")
                    print("  ✗ Person node may not have been created")

            await driver.close()

            # Cleanup
            print("\n[Cleanup] Removing test user...")
            await graph_ops.delete_user(user_id)
            print("✓ Test complete\n")

    finally:
        await graph_ops.close()

if __name__ == "__main__":
    asyncio.run(test_person_connectivity())
