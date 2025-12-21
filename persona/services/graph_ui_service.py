"""
Service for fetching graph UI visualization data with efficient cursor-based pagination.
"""
from typing import Dict, Any, List, Optional
from persona.core.graph_ops import GraphOps
from server.logging_config import get_logger

logger = get_logger(__name__)


class GraphUIService:
    """Service for retrieving graph UI data with efficient pagination."""

    @staticmethod
    async def get_graph_ui_data(
        user_id: str,
        graph_ops: GraphOps,
        book_id: Optional[int] = None,
        highlight_id: Optional[int] = None,
        writing_id: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive graph data for UI visualization with optional filtering and cursor-based pagination.

        Args:
            user_id: The user ID to fetch data for
            graph_ops: Graph operations instance
            book_id: Optional book ID to filter nodes
            highlight_id: Optional highlight ID to filter nodes
            writing_id: Optional writing ID to filter nodes
            date_from: Optional start date (ISO 8601 format, e.g., "2025-02-24") to filter nodes by created_at
            date_to: Optional end date (ISO 8601 format, e.g., "2025-02-26") to filter nodes by created_at
            limit: Optional maximum number of nodes to return (None = all nodes). Uses composite index for efficiency.
            cursor: Optional cursor for pagination (node name to start after). Requires composite index.

        Pagination Strategy:
        - Uses composite index on (UserId, name) for O(log n) seek performance
        - Cursor = last node name from previous page
        - Query: WHERE n.UserId = $user_id AND n.name > $cursor ORDER BY n.name LIMIT $limit
        - Returns pagination metadata: {total_nodes, has_more, next_cursor, returned_nodes}

        Returns:
        - topics: Aggregated by type/discipline with entity and relationship counts
        - insights: High-confidence nodes (confidence >= 0.7)
        - sources: Aggregated by perspective field if available
        - nodes: Filtered graph nodes with properties (paginated if limit specified)
        - relationships: Relationships connecting returned nodes only
        - pagination: Metadata (only if limit specified)
        """
        neo4j_manager = graph_ops.neo4j_manager

        # Log incoming filter parameters
        logger.info(f"GraphUIService.get_graph_ui_data called for user={user_id}")
        logger.info(f"  Filters: book_id={book_id}, highlight_id={highlight_id}, writing_id={writing_id}, date_from={date_from}, date_to={date_to}")
        logger.info(f"  Pagination: limit={limit}, cursor={cursor}")

        # Set default min_confidence
        min_confidence = 0.7

        # Handle date_to: if it's just a date (no time component), append end-of-day time
        processed_date_to = date_to
        if date_to is not None and len(date_to) == 10:  # Format: YYYY-MM-DD
            processed_date_to = f"{date_to}T23:59:59.999999+00:00"

        # Build base parameters
        params = {
            "user_id": user_id,
            "min_confidence": min_confidence,
            "has_book_id": book_id is not None,
            "book_id": book_id if book_id is not None else 0,
            "has_highlight_id": highlight_id is not None,
            "highlight_id": highlight_id if highlight_id is not None else 0,
            "has_writing_id": writing_id is not None,
            "writing_id": writing_id if writing_id is not None else 0,
            "has_date_from": date_from is not None,
            "date_from": date_from if date_from is not None else "",
            "has_date_to": date_to is not None,
            "date_to": processed_date_to if processed_date_to is not None else ""
        }

        # Add pagination parameters
        if limit is not None:
            params["limit"] = limit
            params["has_cursor"] = cursor is not None
            params["cursor"] = cursor if cursor is not None else ""

        logger.info(f"Active filters: {[k for k, v in params.items() if k.startswith('has_') and v]}")

        # Query 1: Get total node count (for pagination metadata)
        # Only run if pagination is enabled
        total_nodes = None
        if limit is not None:
            count_query = """
            MATCH (n:NodeName)
            WHERE n.UserId = $user_id
              AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
              AND (NOT $has_book_id OR $book_id IN n.book_id)
              AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
              AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
              AND (NOT $has_date_from OR n.created_at >= $date_from)
              AND (NOT $has_date_to OR n.created_at <= $date_to)
            RETURN count(n) AS total
            """
            async with neo4j_manager.driver.session() as session:
                logger.info("Executing total count query for pagination...")
                count_result = await session.run(count_query, params)
                count_data = await count_result.data()
                total_nodes = count_data[0]["total"] if count_data else 0
                logger.info(f"Total nodes matching filters: {total_nodes}")

        # Query 2: Get topics (aggregated by discipline)
        # Note: bloom_distribution comes from CognitiveLevel nodes linked to Concepts in this discipline
        topics_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND n.discipline IS NOT NULL
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
          AND (NOT $has_date_from OR n.created_at >= $date_from)
          AND (NOT $has_date_to OR n.created_at <= $date_to)
        WITH n.discipline AS topic,
             count(n) AS entity_count
        
        // Get CognitiveLevel nodes for concepts in this discipline (with filters)
        OPTIONAL MATCH (concept:NodeName)-[:HAS_UNDERSTANDING_LEVEL]->(cl:NodeName)
        WHERE concept.UserId = $user_id
          AND concept.discipline = topic
          AND cl.type = 'CognitiveLevel'
          AND (NOT $has_book_id OR $book_id IN concept.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN concept.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN concept.writing_id)
          AND (NOT $has_date_from OR concept.created_at >= $date_from)
          AND (NOT $has_date_to OR concept.created_at <= $date_to)
        WITH topic, entity_count, collect(cl.name) AS bloom_levels
        
        // Get relationship count for this discipline
        OPTIONAL MATCH (n1:NodeName)-[r]-(n2:NodeName)
        WHERE n1.UserId = $user_id
          AND n1.discipline = topic
          AND (NOT $has_book_id OR $book_id IN n1.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n1.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n1.writing_id)
          AND (NOT $has_date_from OR n1.created_at >= $date_from)
          AND (NOT $has_date_to OR n1.created_at <= $date_to)
        WITH topic, entity_count, bloom_levels, count(DISTINCT r) AS relationship_count
        RETURN topic AS name,
               entity_count,
               relationship_count,
               bloom_levels
        ORDER BY entity_count DESC
        """

        # Query 3: Get insights (high-confidence nodes >= min_confidence)
        insights_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND n.confidence IS NOT NULL
          AND n.confidence >= $min_confidence
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
          AND (NOT $has_date_from OR n.created_at >= $date_from)
          AND (NOT $has_date_to OR n.created_at <= $date_to)
        RETURN n.name AS description,
               n.confidence AS confidence,
               coalesce(n.discipline, n.type, 'Unknown') AS source
        ORDER BY n.confidence DESC
        LIMIT 50
        """

        # Query 4: Get sources - aggregated count by entity type
        sources_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader']
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
          AND (NOT $has_date_from OR n.created_at >= $date_from)
          AND (NOT $has_date_to OR n.created_at <= $date_to)
        WITH
          size([id IN n.book_id WHERE id IS NOT NULL]) AS book_count,
          size([id IN n.highlight_id WHERE id IS NOT NULL]) AS highlight_count,
          size([id IN n.writing_id WHERE id IS NOT NULL]) AS writing_count
        RETURN
          sum(book_count) AS total_book_refs,
          sum(highlight_count) AS total_highlight_refs,
          sum(writing_count) AS total_writing_refs
        """

        # Query 5: Get nodes with cursor-based pagination
        # Uses composite index on (UserId, name) for efficient seeking
        # For Concept nodes, we also fetch their highest CognitiveLevel
        nodes_query_base = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
          AND (NOT $has_date_from OR n.created_at >= $date_from)
          AND (NOT $has_date_to OR n.created_at <= $date_to)
        
        // Get highest cognitive level for Concept nodes
        OPTIONAL MATCH (n)-[:HAS_UNDERSTANDING_LEVEL]->(cl:NodeName)
        WHERE n.type = 'Concept' AND cl.type = 'CognitiveLevel'
        WITH n, 
             CASE 
               WHEN cl.name = 'Create' THEN 6
               WHEN cl.name = 'Evaluate' THEN 5
               WHEN cl.name = 'Analyze' THEN 4
               WHEN cl.name = 'Apply' THEN 3
               WHEN cl.name = 'Understand' THEN 2
               WHEN cl.name = 'Remember' THEN 1
               ELSE 0
             END AS level_rank,
             cl.name AS level_name
        WITH n, 
             max(level_rank) AS max_rank,
             collect(level_name) AS all_levels
        WITH n,
             CASE max_rank
               WHEN 6 THEN 'Create'
               WHEN 5 THEN 'Evaluate'
               WHEN 4 THEN 'Analyze'
               WHEN 3 THEN 'Apply'
               WHEN 2 THEN 'Understand'
               WHEN 1 THEN 'Remember'
               ELSE ''
             END AS highest_bloom_level
        """

        # Add cursor condition if provided (efficient with composite index)
        if limit is not None and cursor:
            nodes_query_base += "  WHERE n.name > $cursor\n"

        nodes_query = nodes_query_base + """
        RETURN elementId(n) AS id,
               n.name AS name,
               n.type AS type,
               n.discipline AS discipline,
               highest_bloom_level AS bloom_level,
               n.confidence AS confidence,
               n.chunk_ids AS chunk_ids,
               n.book_id AS book_id,
               n.highlight_id AS highlight_id,
               n.writing_id AS writing_id,
               n.created_at AS created_at,
               n.concept_uuid AS concept_uuid,
               properties(n) AS properties
        ORDER BY n.name
        """

        # Add LIMIT only if pagination is enabled
        if limit is not None:
            # Fetch one extra node to check if there are more pages
            nodes_query += f"LIMIT {limit + 1}\n"

        # Execute all queries
        async with neo4j_manager.driver.session() as session:
            # Topics
            logger.info("Executing topics query...")
            topics_result = await session.run(topics_query, params)
            topics_data = await topics_result.data()
            logger.info(f"Topics query returned {len(topics_data)} results")

            # Process bloom levels into distribution
            # Map bloom level names to numeric values
            bloom_level_map = {
                "Remember": 1,
                "Understand": 2,
                "Apply": 3,
                "Analyze": 4,
                "Evaluate": 5,
                "Create": 6
            }
            
            topics_list = []
            for record in topics_data:
                bloom_levels = record.get("bloom_levels", [])
                bloom_distribution = {}
                for level in bloom_levels:
                    if level and level in bloom_level_map:
                        numeric_level = bloom_level_map[level]
                        bloom_distribution[numeric_level] = bloom_distribution.get(numeric_level, 0) + 1

                topics_list.append({
                    "name": record["name"],
                    "entity_count": record["entity_count"],
                    "relationship_count": record["relationship_count"],
                    "bloom_distribution": bloom_distribution
                })

            # Insights
            logger.info("Executing insights query...")
            insights_result = await session.run(insights_query, params)
            insights_data = await insights_result.data()
            logger.info(f"Insights query returned {len(insights_data)} results")

            # Sources
            logger.info("Executing sources query...")
            sources_result = await session.run(sources_query, params)
            sources_data = await sources_result.data()
            logger.info(f"Sources query returned {len(sources_data)} results")

            # Nodes (with pagination)
            logger.info("Executing nodes query...")
            nodes_result = await session.run(nodes_query, params)
            nodes_data = await nodes_result.data()
            logger.info(f"Nodes query returned {len(nodes_data)} results")

            # Process pagination metadata
            has_more = False
            next_cursor = None
            if limit is not None and len(nodes_data) > limit:
                has_more = True
                # Remove the extra node we fetched
                nodes_data = nodes_data[:limit]
                # Set next_cursor to the last node's name
                next_cursor = nodes_data[-1]["name"]

            nodes = []
            for node in nodes_data:
                # Remove UserId and other metadata from properties dict
                props = node.get("properties", {})
                props.pop("UserId", None)
                props.pop("name", None)
                props.pop("type", None)
                props.pop("discipline", None)
                props.pop("bloom_level", None)
                props.pop("confidence", None)
                props.pop("chunk_ids", None)
                props.pop("book_id", None)
                props.pop("highlight_id", None)
                props.pop("writing_id", None)
                props.pop("created_at", None)
                props.pop("concept_uuid", None)
                props.pop("embedding", None)

                nodes.append({
                    "id": node["id"],
                    "name": node["name"],
                    "type": node.get("type"),
                    "discipline": node.get("discipline"),
                    "bloom_level": node.get("bloom_level"),
                    "confidence": node.get("confidence"),
                    "chunk_ids": node.get("chunk_ids", []),
                    "book_id": node.get("book_id", []),
                    "highlight_id": node.get("highlight_id", []),
                    "writing_id": node.get("writing_id", []),
                    "created_at": node.get("created_at"),
                    "concept_uuid": node.get("concept_uuid"),
                    "properties": props
                })

        # Query 6: Get relationships (only between returned nodes)
        # Build list of node names for filtering
        node_names = [node["name"] for node in nodes]

        if node_names:
            relationships_query = """
            MATCH (source:NodeName)-[r]->(target:NodeName)
            WHERE source.UserId = $user_id AND target.UserId = $user_id
              AND source.name IN $node_names
              AND target.name IN $node_names
              AND NOT source.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
              AND NOT target.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
            RETURN source.name AS source,
                   target.name AS target,
                   r.value AS relation
            """

            async with neo4j_manager.driver.session() as session:
                logger.info("Executing relationships query...")
                rels_params = {**params, "node_names": node_names}
                rels_result = await session.run(relationships_query, rels_params)
                relationships = await rels_result.data()
                logger.info(f"Relationships query returned {len(relationships)} results")
        else:
            relationships = []

        logger.info(f"Final results summary:")
        logger.info(f"  Topics: {len(topics_list)}, Insights: {len(insights_data)}, Sources: {len(sources_data)}")
        logger.info(f"  Nodes: {len(nodes)}, Relationships: {len(relationships)}")

        # Build response
        response = {
            "topics": topics_list,
            "insights": insights_data,
            "sources": sources_data,
            "nodes": nodes,
            "relationships": relationships
        }

        # Add pagination metadata only if limit was specified
        if limit is not None:
            response["pagination"] = {
                "total_nodes": total_nodes,
                "returned_nodes": len(nodes),
                "has_more": has_more,
                "next_cursor": next_cursor,
                "limit": limit
            }
            logger.info(f"  Pagination: total={total_nodes}, returned={len(nodes)}, has_more={has_more}, next_cursor={next_cursor}")

        return response
