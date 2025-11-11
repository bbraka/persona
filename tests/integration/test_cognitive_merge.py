#!/usr/bin/env python3
"""
Test that cognitive level history is preserved when concepts are merged.

This test verifies:
1. Create Concept A with CognitiveLevel "Remember"
2. Create similar Concept A' with CognitiveLevel "Understand" (should merge)
3. Verify Concept A now has BOTH CognitiveLevel connections (Remember + Understand)
"""

import asyncio
from persona.core.graph_ops import GraphOps
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.models.schema import NodeModel, RelationshipModel, NodesAndRelationshipsResponse
from server.config import config

async def test_cognitive_level_preservation():
    """Test that multiple cognitive levels are preserved when concepts merge"""

    user_id = "test_user_cognitive_merge"

    neo4j_manager = Neo4jConnectionManager()
    graph_ops = GraphOps(neo4j_manager=neo4j_manager)

    try:
        # Initialize
        async with graph_ops:
            # Clean up any existing test data
            try:
                await graph_ops.delete_user(user_id)
            except:
                pass
            await graph_ops.create_user(user_id)

            print("=" * 80)
            print("TEST: Cognitive Level Preservation During Concept Merge")
            print("=" * 80)

            # STEP 1: Create first concept with "Remember" level
            print("\n[Step 1] Creating Concept with 'Remember' cognitive level...")

            concept_uuid_1 = "test-uuid-001"

            concept_1 = NodeModel(
                name="Objectivism is a philosophical system based on rational self-interest",
                type="Concept",
                properties={"discipline": "Philosophy", "concept_uuid": concept_uuid_1}
            )

            cognitive_level_1 = NodeModel(
                name="Remember",
                type="CognitiveLevel",
                properties={"discipline": "Education", "concept_uuid": concept_uuid_1}
            )

            relationship_1 = RelationshipModel(
                source=concept_1.name,
                target=cognitive_level_1.name,
                relation="HAS_UNDERSTANDING_LEVEL"
            )

            graph_update_1 = NodesAndRelationshipsResponse(
                nodes=[concept_1, cognitive_level_1],
                relationships=[relationship_1]
            )

            await graph_ops.update_graph_transactional(graph_update_1, user_id)
            print(f"✓ Created: '{concept_1.name[:60]}...'")
            print(f"  Connected to: CognitiveLevel 'Remember'")

            # STEP 2: Create similar concept with "Understand" level (should merge)
            print("\n[Step 2] Creating similar Concept with 'Understand' cognitive level...")
            print("  (This should merge into the existing concept)")

            concept_uuid_2 = "test-uuid-002"  # Different UUID initially

            # Use EXACT same name to force merge (in real usage, similarity threshold handles this)
            concept_2 = NodeModel(
                name="Objectivism is a philosophical system based on rational self-interest",
                type="Concept",
                properties={"discipline": "Philosophy", "concept_uuid": concept_uuid_2}
            )

            cognitive_level_2 = NodeModel(
                name="Understand",
                type="CognitiveLevel",
                properties={"discipline": "Education", "concept_uuid": concept_uuid_2}
            )

            relationship_2 = RelationshipModel(
                source=concept_2.name,
                target=cognitive_level_2.name,
                relation="HAS_UNDERSTANDING_LEVEL"
            )

            graph_update_2 = NodesAndRelationshipsResponse(
                nodes=[concept_2, cognitive_level_2],
                relationships=[relationship_2]
            )

            await graph_ops.update_graph_transactional(graph_update_2, user_id)
            print(f"✓ Attempted to create: '{concept_2.name[:60]}...'")
            print(f"  Expected to merge with existing concept")

            # STEP 3: Verify results
            print("\n[Step 3] Verifying cognitive level preservation...")

            # Query for concepts with multiple cognitive levels
            from neo4j import AsyncGraphDatabase
            driver = AsyncGraphDatabase.driver(config.NEO4J.URI, auth=(config.NEO4J.USER, config.NEO4J.PASSWORD))

            async with driver.session() as session:
                # Find all Concept nodes for this user
                result = await session.run("""
                    MATCH (concept:NodeName {type: 'Concept', UserId: $user_id})
                    RETURN concept.name as name
                """, user_id=user_id)

                concepts = [record["name"] async for record in result]
                print(f"\n  Total concepts created: {len(concepts)}")
                if len(concepts) == 1:
                    print(f"  ✓ Deduplication worked - only 1 concept exists")
                else:
                    print(f"  ✗ Expected 1 concept, found {len(concepts)}")
                    for c in concepts:
                        print(f"    - {c[:60]}...")

                # Find cognitive levels for each concept
                for concept_name in concepts:
                    result = await session.run("""
                        MATCH (concept:NodeName {name: $concept_name, UserId: $user_id})-[r:HAS_UNDERSTANDING_LEVEL]->(cl:NodeName {type: 'CognitiveLevel'})
                        RETURN cl.name as level, cl.concept_uuid as cl_uuid, concept.concept_uuid as concept_uuid
                        ORDER BY cl.name
                    """, concept_name=concept_name, user_id=user_id)

                    levels = []
                    async for record in result:
                        levels.append({
                            "level": record["level"],
                            "cl_uuid": record["cl_uuid"],
                            "concept_uuid": record["concept_uuid"]
                        })

                    print(f"\n  Concept: '{concept_name[:60]}...'")
                    print(f"  Cognitive Levels connected: {len(levels)}")

                    for level_info in levels:
                        print(f"    - {level_info['level']}")
                        print(f"      CognitiveLevel UUID: {level_info['cl_uuid']}")
                        print(f"      Concept UUID: {level_info['concept_uuid']}")

                    # Verify results
                    if len(levels) == 2:
                        level_names = {l["level"] for l in levels}
                        if level_names == {"Remember", "Understand"}:
                            print("\n  ✓✓✓ SUCCESS! Both cognitive levels preserved!")
                            print("  ✓ History of cognitive progression is intact")

                            # Check UUIDs match
                            uuids = {l["concept_uuid"] for l in levels}
                            if len(uuids) == 1:
                                print(f"  ✓ Both CognitiveLevels have matching concept_uuid: {list(uuids)[0]}")
                            else:
                                print(f"  ✗ WARNING: CognitiveLevels have different concept_uuids: {uuids}")
                        else:
                            print(f"\n  ✗ Wrong cognitive levels: {level_names}")
                    else:
                        print(f"\n  ✗✗✗ FAILURE! Expected 2 cognitive levels, found {len(levels)}")
                        print("  ✗ Cognitive level history was NOT preserved")

            await driver.close()

            # Cleanup
            print("\n[Cleanup] Removing test user...")
            await graph_ops.delete_user(user_id)
            print("✓ Test complete\n")

    finally:
        await graph_ops.close()

if __name__ == "__main__":
    asyncio.run(test_cognitive_level_preservation())
