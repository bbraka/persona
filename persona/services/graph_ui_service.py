"""
Service for fetching graph UI visualization data.
"""
from typing import Dict, Any, List, Optional
from persona.core.graph_ops import GraphOps
from server.logging_config import get_logger

logger = get_logger(__name__)


class GraphUIService:
    """Service for retrieving graph UI data."""

    @staticmethod
    async def get_graph_ui_data(
        user_id: str,
        graph_ops: GraphOps,
        book_id: Optional[int] = None,
        highlight_id: Optional[int] = None,
        writing_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive graph data for UI visualization with optional filtering by entity IDs.

        Args:
            user_id: The user ID to fetch data for
            graph_ops: Graph operations instance
            book_id: Optional book ID to filter nodes
            highlight_id: Optional highlight ID to filter nodes
            writing_id: Optional writing ID to filter nodes

        Returns raw Neo4j response dictionaries containing:
        - topics: Aggregated by type/discipline with entity and relationship counts
        - insights: High-confidence nodes (confidence >= 0.7)
        - sources: Aggregated by perspective field if available
        - nodes: Filtered graph nodes with properties
        - relationships: Relationships connecting filtered nodes
        """
        neo4j_manager = graph_ops.neo4j_manager

        # Log incoming filter parameters
        logger.info(f"GraphUIService.get_graph_ui_data called for user={user_id}")
        logger.info(f"  Filters: book_id={book_id}, highlight_id={highlight_id}, writing_id={writing_id}")

        # Set default min_confidence
        min_confidence = 0.7

        # Query 1: Get topics (aggregated by discipline)
        # Filter by entity IDs if provided
        topics_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND n.discipline IS NOT NULL
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
        WITH n.discipline AS topic,
             count(n) AS entity_count,
             collect(n.bloom_level) AS bloom_levels
        OPTIONAL MATCH (n1:NodeName)-[r]-(n2:NodeName)
        WHERE n1.UserId = $user_id
          AND n1.discipline = topic
          AND (NOT $has_book_id OR $book_id IN n1.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n1.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n1.writing_id)
        WITH topic, entity_count, bloom_levels, count(DISTINCT r) AS relationship_count
        RETURN topic AS name,
               entity_count,
               relationship_count,
               bloom_levels
        ORDER BY entity_count DESC
        """

        # Query 2: Get insights (high-confidence nodes >= min_confidence)
        insights_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND n.confidence IS NOT NULL
          AND n.confidence >= $min_confidence
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
        RETURN n.name AS description,
               n.confidence AS confidence,
               coalesce(n.discipline, n.type, 'Unknown') AS source
        ORDER BY n.confidence DESC
        LIMIT 50
        """

        # Query 3: Get sources - aggregated count by entity type
        # Returns counts of nodes by book_id, highlight_id, writing_id
        sources_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader']
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
        WITH
          size([id IN n.book_id WHERE id IS NOT NULL]) AS book_count,
          size([id IN n.highlight_id WHERE id IS NOT NULL]) AS highlight_count,
          size([id IN n.writing_id WHERE id IS NOT NULL]) AS writing_count
        RETURN
          sum(book_count) AS total_book_refs,
          sum(highlight_count) AS total_highlight_refs,
          sum(writing_count) AS total_writing_refs
        """

        # Query 4: Get all nodes with their properties
        # Exclude internal node types used for tracking (Reading Session, etc.)
        # Filter by entity IDs if provided
        nodes_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
          AND (NOT $has_book_id OR $book_id IN n.book_id)
          AND (NOT $has_highlight_id OR $highlight_id IN n.highlight_id)
          AND (NOT $has_writing_id OR $writing_id IN n.writing_id)
        RETURN elementId(n) AS id,
               n.name AS name,
               n.type AS type,
               n.discipline AS discipline,
               n.bloom_level AS bloom_level,
               n.confidence AS confidence,
               n.chunk_ids AS chunk_ids,
               n.book_id AS book_id,
               n.highlight_id AS highlight_id,
               n.writing_id AS writing_id,
               properties(n) AS properties
        ORDER BY n.name
        """

        # Query 5: Get all relationships
        # Relationships are stored with the relation type as the relationship type
        # and r.value contains the relation name
        # Exclude relationships involving internal node types and Book nodes
        # Filter to only show relationships between nodes that match the filter criteria
        relationships_query = """
        MATCH (source:NodeName)-[r]->(target:NodeName)
        WHERE source.UserId = $user_id AND target.UserId = $user_id
          AND NOT source.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
          AND NOT target.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader', 'Book']
          AND (NOT $has_book_id OR ($book_id IN source.book_id AND $book_id IN target.book_id))
          AND (NOT $has_highlight_id OR ($highlight_id IN source.highlight_id AND $highlight_id IN target.highlight_id))
          AND (NOT $has_writing_id OR ($writing_id IN source.writing_id AND $writing_id IN target.writing_id))
        RETURN source.name AS source,
               target.name AS target,
               r.value AS relation
        """

        # Build parameters dictionary with boolean flags for conditional filtering
        # This prevents Cypher injection by using parameterized queries
        params = {
            "user_id": user_id,
            "min_confidence": min_confidence,
            "has_book_id": book_id is not None,
            "book_id": book_id if book_id is not None else 0,
            "has_highlight_id": highlight_id is not None,
            "highlight_id": highlight_id if highlight_id is not None else 0,
            "has_writing_id": writing_id is not None,
            "writing_id": writing_id if writing_id is not None else 0
        }

        logger.info(f"Query parameters prepared:")
        logger.info(f"  has_book_id={params['has_book_id']}, book_id={params['book_id']}")
        logger.info(f"  has_highlight_id={params['has_highlight_id']}, highlight_id={params['highlight_id']}")
        logger.info(f"  has_writing_id={params['has_writing_id']}, writing_id={params['writing_id']}")
        logger.info(f"  min_confidence={params['min_confidence']}")

        # Log which filters are active
        active_filters = []
        if params['has_book_id']:
            active_filters.append(f"book_id={params['book_id']}")
        if params['has_highlight_id']:
            active_filters.append(f"highlight_id={params['highlight_id']}")
        if params['has_writing_id']:
            active_filters.append(f"writing_id={params['writing_id']}")

        if active_filters:
            logger.info(f"Active filters: {', '.join(active_filters)}")
        else:
            logger.info("No filters active - returning all data")

        # Execute all queries
        async with neo4j_manager.driver.session() as session:
            # Topics
            logger.info("Executing topics query...")
            topics_result = await session.run(topics_query, params)
            topics_data = await topics_result.data()
            logger.info(f"Topics query returned {len(topics_data)} results")

            # Process bloom levels into distribution
            topics_list = []
            for record in topics_data:
                bloom_levels = record.get("bloom_levels", [])
                bloom_distribution = {}
                for level in bloom_levels:
                    if level:
                        bloom_distribution[level] = bloom_distribution.get(level, 0) + 1

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

            # Nodes - clean up the properties to avoid duplication
            logger.info("Executing nodes query...")
            nodes_result = await session.run(nodes_query, params)
            nodes_data = await nodes_result.data()
            logger.info(f"Nodes query returned {len(nodes_data)} results")

            nodes = []
            for node in nodes_data:
                # Remove UserId and other metadata from properties dict
                props = node.get("properties", {})
                props.pop("UserId", None)
                props.pop("name", None)  # Already in top-level
                props.pop("type", None)  # Already in top-level

                # Remove PKG properties from props dict since they're in top-level
                props.pop("discipline", None)
                props.pop("bloom_level", None)
                props.pop("confidence", None)
                props.pop("chunk_ids", None)  # Already in top-level

                # Remove entity ID arrays from props dict since they're in top-level
                props.pop("book_id", None)
                props.pop("highlight_id", None)
                props.pop("writing_id", None)

                # Remove embedding vector (never send to client)
                props.pop("embedding", None)

                nodes.append({
                    "id": node["id"],
                    "name": node["name"],
                    "type": node.get("type"),
                    "discipline": node.get("discipline"),
                    "bloom_level": node.get("bloom_level"),
                    "confidence": node.get("confidence"),
                    "chunk_ids": node.get("chunk_ids", []),  # Array of chunk UUIDs
                    "book_id": node.get("book_id", []),  # Array of book IDs
                    "highlight_id": node.get("highlight_id", []),  # Array of highlight IDs
                    "writing_id": node.get("writing_id", []),  # Array of writing IDs
                    "properties": props  # Only custom properties remain
                })

            # Relationships
            logger.info("Executing relationships query...")
            rels_result = await session.run(relationships_query, params)
            relationships = await rels_result.data()
            logger.info(f"Relationships query returned {len(relationships)} results")

        logger.info(f"Final results summary:")
        logger.info(f"  Topics: {len(topics_list)}, Insights: {len(insights_data)}, Sources: {len(sources_data)}")
        logger.info(f"  Nodes: {len(nodes)}, Relationships: {len(relationships)}")

        return {
            "topics": topics_list,
            "insights": insights_data,
            "sources": sources_data,
            "nodes": nodes,
            "relationships": relationships
        }
