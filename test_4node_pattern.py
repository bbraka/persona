#!/usr/bin/env python3
"""
Test script to verify the new 4-node pattern for highlight/note ingestion.
This tests that the system creates:
1. Highlight node
2. UserNote node
3. Concept node (synthesized statement)
4. CognitiveLevel node
With appropriate relationships between them.
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_4node_pattern"

async def test_4node_pattern():
    """Test the 4-node pattern with a highlight and note"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 80)
        print("Testing 4-Node Pattern for Highlight/Note Ingestion")
        print("=" * 80)

        # Step 1: Create test user
        print("\n1. Creating test user...")
        response = await client.post(f"{BASE_URL}/users/{TEST_USER_ID}")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

        # Step 2: Delete any existing data
        print("\n2. Cleaning up existing data...")
        response = await client.delete(f"{BASE_URL}/users/{TEST_USER_ID}")
        print(f"   Status: {response.status_code}")

        # Step 3: Recreate user
        print("\n3. Recreating user...")
        response = await client.post(f"{BASE_URL}/users/{TEST_USER_ID}")
        print(f"   Status: {response.status_code}")

        # Step 4: Ingest a highlight with a note
        print("\n4. Ingesting highlight with note...")
        test_data = {
            "title": "Test: Frank Knight and Henry Simons",
            "content": "Highlight: Frank Knight, Henry Simons\n\nNote: Milton Friedman's advisors at the University of Chicago",
            "chunk_ids": [],
            "metadata": {
                "book_id": "999",
                "highlight_id": "777",
                "date": datetime.utcnow().isoformat()
            }
        }

        response = await client.post(
            f"{BASE_URL}/users/{TEST_USER_ID}/ingest",
            json=test_data
        )
        print(f"   Status: {response.status_code}")
        if response.status_code != 201:
            print(f"   Error: {response.text}")
            return

        result = response.json()
        print(f"   Nodes created: {result.get('nodes_created', 0)}")
        print(f"   Relationships created: {result.get('relationships_created', 0)}")

        # Step 5: Wait a bit for processing
        print("\n5. Waiting for graph processing...")
        await asyncio.sleep(3)

        # Step 6: Retrieve all nodes
        print("\n6. Retrieving all nodes...")
        response = await client.get(f"{BASE_URL}/users/{TEST_USER_ID}/graph")
        graph_data = response.json()

        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])

        print(f"\n   Total nodes: {len(nodes)}")
        print(f"   Total relationships: {len(relationships)}")

        # Step 7: Analyze nodes by type
        print("\n7. Analyzing node types...")
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "Unknown")
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)

        print("\n   Nodes by type:")
        for node_type, nodes_of_type in sorted(node_types.items()):
            print(f"\n   {node_type} ({len(nodes_of_type)} nodes):")
            for node in nodes_of_type:
                print(f"      - {node.get('name')}")

        # Step 8: Check for 4-node pattern
        print("\n8. Checking for 4-node pattern...")
        has_highlight = "Highlight" in node_types
        has_usernote = "UserNote" in node_types
        has_concept = "Concept" in node_types
        has_cognitive_level = "CognitiveLevel" in node_types

        print(f"   ✓ Highlight node: {has_highlight}")
        print(f"   ✓ UserNote node: {has_usernote}")
        print(f"   ✓ Concept node: {has_concept}")
        print(f"   ✓ CognitiveLevel node: {has_cognitive_level}")

        # Step 9: Check relationships
        print("\n9. Analyzing relationships...")
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("relation", "Unknown")
            if rel_type not in rel_types:
                rel_types[rel_type] = []
            rel_types[rel_type].append(rel)

        print("\n   Relationships by type:")
        for rel_type, rels_of_type in sorted(rel_types.items()):
            print(f"\n   {rel_type} ({len(rels_of_type)}):")
            for rel in rels_of_type:
                print(f"      {rel.get('source')} → {rel.get('target')}")

        # Step 10: Verify expected relationships
        print("\n10. Verifying expected relationship patterns...")
        has_annotated_with = "ANNOTATED_WITH" in rel_types
        has_synthesized_into = "SYNTHESIZED_INTO" in rel_types
        has_understanding_level = "HAS_UNDERSTANDING_LEVEL" in rel_types

        print(f"   ✓ ANNOTATED_WITH: {has_annotated_with}")
        print(f"   ✓ SYNTHESIZED_INTO: {has_synthesized_into}")
        print(f"   ✓ HAS_UNDERSTANDING_LEVEL: {has_understanding_level}")

        # Step 11: Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        all_nodes_present = has_highlight and has_usernote and has_concept and has_cognitive_level
        all_rels_present = has_annotated_with and has_synthesized_into and has_understanding_level

        if all_nodes_present and all_rels_present:
            print("✅ SUCCESS: 4-node pattern is working correctly!")
            print(f"   - All 4 node types created")
            print(f"   - All expected relationships present")
        else:
            print("❌ FAILURE: 4-node pattern incomplete")
            if not all_nodes_present:
                print(f"   - Missing node types")
            if not all_rels_present:
                print(f"   - Missing relationship types")

        print("\n" + "=" * 80)

        # Step 12: Detailed node data
        if has_concept and node_types.get("Concept"):
            print("\nDetailed Concept Node:")
            for concept in node_types["Concept"]:
                print(f"   Name: {concept.get('name')}")
                print(f"   Type: {concept.get('type')}")
                print(f"   Properties: {json.dumps(concept.get('properties', {}), indent=6)}")

        if has_cognitive_level and node_types.get("CognitiveLevel"):
            print("\nDetailed CognitiveLevel Node:")
            for cog_level in node_types["CognitiveLevel"]:
                print(f"   Name: {cog_level.get('name')}")
                print(f"   Type: {cog_level.get('type')}")
                print(f"   Properties: {json.dumps(cog_level.get('properties', {}), indent=6)}")

if __name__ == "__main__":
    asyncio.run(test_4node_pattern())
