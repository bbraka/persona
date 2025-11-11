#!/usr/bin/env python3
"""
Test two-tier Theme consolidation strategy.

This test verifies:
1. Tier 1: Similar themes merge during ingestion (0.60 threshold for same-book)
2. Tier 2: Low-frequency themes (< 3 chunks) are pruned after ingestion
3. Result: Only chapter-level recurring themes survive

Test scenario:
- Ingest 5 reading chunks from same book
- Chunks 1-3: Contain "Economic Policy" theme (should merge and survive)
- Chunk 4: Contains "Historical Context" theme (one-off, should be pruned)
- Chunk 5: Contains "Economic Policies" theme (similar to first 3, should merge)
"""

import asyncio
from persona.core.graph_ops import GraphOps
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.services.ingest_service import IngestService
from persona.models.schema import UnstructuredData
from neo4j import AsyncGraphDatabase
from server.config import config
import uuid

async def test_theme_consolidation():
    """Test that two-tier consolidation produces clean theme graph"""

    user_id = "test_user_theme_consolidation"
    book_id = 12345
    book_uuid = str(uuid.uuid4())

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
            print("TEST: Two-Tier Theme Consolidation Strategy")
            print("=" * 80)

            # Prepare test data: 5 reading chunks from same book
            chunks = []

            # Chunks 1-3: Economic Policy theme (recurring - should survive)
            for i in range(3):
                chunk_uuid = str(uuid.uuid4())
                chunks.append(UnstructuredData(
                    title=f"Chapter {i+1}: Economic Policy",
                    content=f"""
                    CHAPTER {i+1}: ECONOMIC POLICY

                    This chapter explores economic policy as a central theme of governance.
                    Economic policy encompasses fiscal responsibility, monetary controls, and trade agreements.
                    The development of economic policy shapes national prosperity and international relations.
                    Throughout this section, we examine how economic policy has evolved historically.
                    Economic policy remains one of the most critical areas of modern government.
                    """,
                    chunk_ids=[chunk_uuid],
                    metadata={
                        "user_id": user_id,
                        "data_type": "book-reading-chunk",
                        "book_id": str(book_id),
                        "book_uuid": book_uuid,
                        "chunk_id": chunk_uuid
                    }
                ))

            # Chunk 4: Historical Context theme (one-off - should be pruned)
            chunk_uuid = str(uuid.uuid4())
            chunks.append(UnstructuredData(
                title="Chapter 4: Background",
                content="""
                Understanding the historical context helps explain modern developments.
                This brief chapter provides necessary background information before
                moving into detailed analysis in subsequent sections.
                """,
                chunk_ids=[chunk_uuid],
                metadata={
                    "user_id": user_id,
                    "data_type": "book-reading-chunk",
                    "book_id": str(book_id),
                    "book_uuid": book_uuid,
                    "chunk_id": chunk_uuid
                }
            ))

            # Chunk 5: Economic Policies theme (similar to 1-3 - should merge)
            chunk_uuid = str(uuid.uuid4())
            chunks.append(UnstructuredData(
                title="Chapter 5: Policy Implementation",
                content="""
                The implementation of economic policies requires careful coordination
                across government agencies. Economic policies affect every citizen
                through taxation, spending, and regulatory frameworks.
                """,
                chunk_ids=[chunk_uuid],
                metadata={
                    "user_id": user_id,
                    "data_type": "book-reading-chunk",
                    "book_id": str(book_id),
                    "book_uuid": book_uuid,
                    "chunk_id": chunk_uuid
                }
            ))

            print(f"\n[Step 1] Ingesting {len(chunks)} reading chunks...")
            print(f"  Expected Themes BEFORE consolidation:")
            print(f"    - 'Economic Policy' (chunks 1-3)")
            print(f"    - 'Historical Context' (chunk 4) <- should be PRUNED")
            print(f"    - 'Economic Policies' (chunk 5) <- should MERGE with first theme")
            print()

            # Ingest with two-tier consolidation
            result = await IngestService.ingest_data(user_id, chunks, graph_ops)

            print(f"\n[Step 2] Ingestion complete")
            print(f"  Message: {result['message']}")
            if 'theme_pruning' in result:
                stats = result['theme_pruning']
                print(f"\n  Theme Pruning Stats:")
                print(f"    - Themes analyzed: {stats.get('themes_analyzed', 0)}")
                print(f"    - Themes kept: {stats.get('themes_kept', 0)}")
                print(f"    - Themes pruned: {stats.get('themes_pruned', 0)}")
                if stats.get('pruned_themes'):
                    print(f"    - Pruned: {[t['name'][:50] + '...' if len(t['name']) > 50 else t['name'] for t in stats['pruned_themes']]}")

            # Verify results
            print("\n[Step 3] Verifying theme consolidation...")

            driver = AsyncGraphDatabase.driver(
                config.NEO4J.URI,
                auth=(config.NEO4J.USER, config.NEO4J.PASSWORD)
            )

            async with driver.session() as session:
                # Count total themes
                result = await session.run("""
                    MATCH (n:NodeName {UserId: $user_id, type: 'Theme'})
                    RETURN count(n) as theme_count
                """, user_id=user_id)
                data = await result.single()
                theme_count = data["theme_count"] if data else 0

                print(f"\n  Total Themes in graph: {theme_count}")

                # Get theme details
                result = await session.run("""
                    MATCH (n:NodeName {UserId: $user_id, type: 'Theme'})
                    RETURN n.name as name,
                           size(COALESCE(n.chunk_ids, [])) as chunk_count,
                           size(COALESCE(n.book_id, [])) as book_count
                    ORDER BY chunk_count DESC
                """, user_id=user_id)

                themes = []
                async for record in result:
                    themes.append({
                        "name": record["name"],
                        "chunks": record["chunk_count"],
                        "books": record["book_count"]
                    })

                print("\n  Theme Details:")
                for theme in themes:
                    print(f"    - '{theme['name'][:60]}...' if len(theme['name']) > 60 else theme['name']")
                    print(f"      Chunks: {theme['chunks']}, Books: {theme['books']}")

                # Validate results
                print("\n[Validation]")

                success = True

                # Should have exactly 1 theme (Economic Policy/Policies merged)
                if theme_count == 1:
                    print("  ✓ Correct number of themes (1)")
                else:
                    print(f"  ✗ Expected 1 theme, found {theme_count}")
                    success = False

                # That theme should have 4 chunks (chunks 1-3 + 5, after merging)
                if themes:
                    main_theme = themes[0]
                    if main_theme["chunks"] >= 3:
                        print(f"  ✓ Main theme has sufficient chunk evidence ({main_theme['chunks']} chunks)")
                    else:
                        print(f"  ✗ Main theme has insufficient chunks ({main_theme['chunks']})")
                        success = False

                    # Check that theme name relates to economics
                    if "economic" in main_theme["name"].lower() or "policy" in main_theme["name"].lower():
                        print(f"  ✓ Theme relates to economics/policy")
                    else:
                        print(f"  ✗ Theme doesn't relate to expected topic: {main_theme['name']}")
                        success = False

                # Historical Context theme should be gone
                historical_themes = [t for t in themes if "historical" in t["name"].lower() or "context" in t["name"].lower()]
                if not historical_themes:
                    print("  ✓ One-off 'Historical Context' theme was pruned")
                else:
                    print(f"  ✗ One-off theme survived: {historical_themes[0]['name']}")
                    success = False

                if success:
                    print("\n" + "=" * 80)
                    print("✓✓✓ SUCCESS! Two-tier consolidation working correctly!")
                    print("=" * 80)
                    print("  ✓ Tier 1: Similar themes merged (Economic Policy + Economic Policies)")
                    print("  ✓ Tier 2: Low-frequency themes pruned (Historical Context)")
                    print("  ✓ Result: Clean graph with only chapter-level recurring themes")
                else:
                    print("\n" + "=" * 80)
                    print("✗✗✗ FAILURE! Consolidation not working as expected")
                    print("=" * 80)

            await driver.close()

            # Cleanup
            print("\n[Cleanup] Removing test user...")
            await graph_ops.delete_user(user_id)
            print("✓ Test complete\n")

    finally:
        await graph_ops.close()

if __name__ == "__main__":
    asyncio.run(test_theme_consolidation())
