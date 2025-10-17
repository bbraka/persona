"""
Service for fetching graph UI visualization data.
"""
from typing import Dict, Any, List
from persona.core.graph_ops import GraphOps


class GraphUIService:
    """Service for retrieving graph UI data."""

    @staticmethod
    async def get_graph_ui_data(user_id: str, graph_ops: GraphOps) -> Dict[str, Any]:
        """
        Fetch comprehensive graph data for UI visualization.

        Returns raw Neo4j response dictionaries containing:
        - topics: Aggregated by type/discipline with entity and relationship counts
        - insights: High-confidence nodes (confidence >= 0.7)
        - sources: Aggregated by perspective field if available
        - nodes: All graph nodes with properties
        - relationships: All graph relationships
        """
        neo4j_manager = graph_ops.neo4j_manager

        # Query 1: Get topics (aggregated by discipline)
        # Discipline is the primary categorization field per the prompts
        topics_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id AND n.discipline IS NOT NULL
        WITH n.discipline AS topic,
             count(n) AS entity_count,
             collect(n.bloom_level) AS bloom_levels
        OPTIONAL MATCH (n1:NodeName)-[r]-(n2:NodeName)
        WHERE n1.UserId = $user_id AND n1.discipline = topic
        WITH topic, entity_count, bloom_levels, count(DISTINCT r) AS relationship_count
        RETURN topic AS name,
               entity_count,
               relationship_count,
               bloom_levels
        ORDER BY entity_count DESC
        """

        # Query 2: Get insights (high-confidence nodes >= 0.7)
        insights_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND n.confidence IS NOT NULL
          AND n.confidence >= 0.7
        RETURN n.name AS description,
               n.confidence AS confidence,
               coalesce(n.discipline, n.type, 'Unknown') AS source
        ORDER BY n.confidence DESC
        LIMIT 50
        """

        # Query 3: Get sources (aggregated by perspective field from custom data)
        # Perspective is used in custom data, type for ingested data
        sources_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
        WITH coalesce(n.perspective, n.type, 'Direct input') AS source_name,
             count(n) AS count
        RETURN source_name AS name,
               count
        ORDER BY count DESC
        """

        # Query 4: Get all nodes with their properties
        # Exclude internal node types used for tracking (Reading Session, etc.)
        nodes_query = """
        MATCH (n:NodeName)
        WHERE n.UserId = $user_id
          AND NOT n.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader']
        RETURN elementId(n) AS id,
               n.name AS name,
               n.type AS type,
               n.discipline AS discipline,
               n.bloom_level AS bloom_level,
               n.confidence AS confidence,
               n.chunk_id AS chunk_id,
               properties(n) AS properties
        ORDER BY n.name
        """

        # Query 5: Get all relationships
        # Relationships are stored with the relation type as the relationship type
        # and r.value contains the relation name
        # Exclude relationships involving internal node types
        relationships_query = """
        MATCH (source:NodeName)-[r]->(target:NodeName)
        WHERE source.UserId = $user_id AND target.UserId = $user_id
          AND NOT source.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader']
          AND NOT target.type IN ['Reading Session', 'ReadingSession', 'CommunityHeader', 'CommunitySubheader']
        RETURN source.name AS source,
               target.name AS target,
               r.value AS relation
        """

        # Execute all queries
        async with neo4j_manager.driver.session() as session:
            # Topics
            topics_result = await session.run(topics_query, {"user_id": user_id})
            topics_data = await topics_result.data()

            # Process bloom levels into distribution
            topics = []
            for record in topics_data:
                bloom_levels = record.get("bloom_levels", [])
                bloom_distribution = {}
                for level in bloom_levels:
                    if level:
                        bloom_distribution[level] = bloom_distribution.get(level, 0) + 1

                topics.append({
                    "name": record["name"],
                    "entity_count": record["entity_count"],
                    "relationship_count": record["relationship_count"],
                    "bloom_distribution": bloom_distribution
                })

            # Insights
            insights_result = await session.run(insights_query, {"user_id": user_id})
            insights = await insights_result.data()

            # Sources
            sources_result = await session.run(sources_query, {"user_id": user_id})
            sources = await sources_result.data()

            # Nodes - clean up the properties to avoid duplication
            nodes_result = await session.run(nodes_query, {"user_id": user_id})
            nodes_data = await nodes_result.data()

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
                props.pop("chunk_id", None)  # Already in top-level

                # Remove embedding vector (never send to client)
                props.pop("embedding", None)

                nodes.append({
                    "id": node["id"],
                    "name": node["name"],
                    "type": node.get("type"),
                    "discipline": node.get("discipline"),
                    "bloom_level": node.get("bloom_level"),
                    "confidence": node.get("confidence"),
                    "chunk_id": node.get("chunk_id"),
                    "properties": props  # Only custom properties remain
                })

            # Relationships
            rels_result = await session.run(relationships_query, {"user_id": user_id})
            relationships = await rels_result.data()

        return {
            "topics": topics,
            "insights": insights,
            "sources": sources,
            "nodes": nodes,
            "relationships": relationships
        }
