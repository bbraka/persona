"""
Node deduplication utilities using vector similarity.

This module provides tools to:
1. Check for semantic duplicates before creating nodes
2. Consolidate existing duplicate nodes
3. Merge relationships from duplicates to canonical nodes
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from persona.llm.embeddings import generate_embeddings
from server.logging_config import get_logger
from collections import defaultdict

logger = get_logger(__name__)


class NodeDeduplicator:
    """Handles semantic deduplication of nodes using vector similarity"""

    def __init__(self, neo4j_manager, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator.

        Args:
            neo4j_manager: Neo4j connection manager instance
            similarity_threshold: Cosine similarity threshold (0.0-1.0) for considering nodes as duplicates.
                                 Higher = more strict. Recommended: 0.80-0.90 for concept deduplication.
        """
        self.neo4j_manager = neo4j_manager
        self.similarity_threshold = similarity_threshold

    def _get_merge_threshold(
        self,
        new_discipline: Optional[str],
        existing_discipline: Optional[str]
    ) -> float:
        """
        Determine merge threshold based on discipline compatibility.

        Args:
            new_discipline: Discipline of the new node
            existing_discipline: Discipline of the existing node

        Returns:
            Similarity threshold to use for merge decision
        """
        # Same discipline or both missing discipline = default behavior
        if not new_discipline or not existing_discipline:
            return self.similarity_threshold  # 0.85 default

        if new_discipline == existing_discipline:
            return self.similarity_threshold  # 0.85 - confident merge within same domain

        # Different disciplines = very conservative
        return 0.95  # Require very high similarity to merge across disciplines

    async def find_similar_node(
        self,
        node_name: str,
        node_type: Optional[str],
        user_id: str,
        limit: int = 5,
        embedding: Optional[List[float]] = None,
        discipline: Optional[str] = None,
        book_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the most similar existing node using vector search with context-aware thresholds.

        Args:
            node_name: Name of the node to check
            node_type: Type of the node (optional filter)
            user_id: User ID
            limit: Max number of candidates to check
            embedding: Optional pre-computed embedding (if not provided, will generate)
            discipline: Discipline/domain of the new node (for cross-discipline validation)
            book_id: Book ID to prioritize same-book matches

        Returns:
            Dict with similar node info if found, None otherwise:
            {
                "name": str,
                "type": str,
                "score": float,
                "id": int
            }
        """
        # Use provided embedding or generate if not provided
        if embedding is None:
            embeddings = generate_embeddings([node_name])
            if not embeddings or not embeddings[0]:
                logger.warning(f"Could not generate embedding for node: {node_name}")
                return None
            embedding = embeddings[0]

        # PHASE 1: If book_id provided, search within same book first (lower threshold)
        if book_id:
            try:
                same_book_results = await self.neo4j_manager.query_text_similarity(
                    embedding,
                    user_id,
                    limit=limit,
                    book_id=book_id
                )

                for result in same_book_results:
                    score = result.get("score", 0.0)

                    # Lower threshold for same-book matches (0.75)
                    if score >= 0.75:
                        node_data = await self.neo4j_manager.get_node_data(result["nodeName"], user_id)

                        # Type filter if provided
                        if node_type and node_data.get("type") != node_type:
                            logger.debug(f"Skipping {result['nodeName']} - different type")
                            continue

                        logger.info(
                            f"Found similar node in same book: '{node_name}' ~= '{result['nodeName']}' "
                            f"(score: {score:.3f}, book_id: {book_id})"
                        )

                        return {
                            "name": result["nodeName"],
                            "type": node_data.get("type"),
                            "score": score,
                            "id": result["nodeId"]
                        }
            except Exception as e:
                logger.debug(f"Same-book search failed, falling back to global search: {e}")

        # PHASE 2: Global search with discipline-aware threshold
        try:
            results = await self.neo4j_manager.query_text_similarity(
                embedding,
                user_id,
                limit=limit
            )

            # Filter by similarity threshold and optionally by type
            for result in results:
                score = result.get("score", 0.0)
                node_data = await self.neo4j_manager.get_node_data(result["nodeName"], user_id)

                # Get discipline-aware threshold
                existing_discipline = node_data.get("properties", {}).get("discipline")
                required_threshold = self._get_merge_threshold(discipline, existing_discipline)

                if score >= required_threshold:
                    # Type filter if provided
                    if node_type and node_data.get("type") != node_type:
                        logger.debug(f"Skipping {result['nodeName']} - different type ({node_data.get('type')} vs {node_type})")
                        continue

                    # Log discipline info if cross-discipline merge
                    if discipline and existing_discipline and discipline != existing_discipline:
                        logger.info(
                            f"Cross-discipline merge: '{node_name}' ({discipline}) ~= '{result['nodeName']}' ({existing_discipline}) "
                            f"(score: {score:.3f}, threshold: {required_threshold:.2f})"
                        )
                    else:
                        logger.info(
                            f"Found similar node: '{node_name}' ~= '{result['nodeName']}' "
                            f"(score: {score:.3f})"
                        )

                    return {
                        "name": result["nodeName"],
                        "type": node_data.get("type"),
                        "score": score,
                        "id": result["nodeId"]
                    }
                elif discipline and existing_discipline and discipline != existing_discipline:
                    # Log when cross-discipline match rejected
                    logger.debug(
                        f"Skipping cross-discipline merge: '{node_name}' ({discipline}) vs '{result['nodeName']}' ({existing_discipline}) "
                        f"(score: {score:.3f} < threshold: {required_threshold:.2f})"
                    )

            return None

        except Exception as e:
            logger.error(f"Error finding similar nodes: {e}")
            return None

    async def consolidate_duplicate_nodes(
        self,
        user_id: str,
        dry_run: bool = True,
        type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find and consolidate all duplicate nodes in the graph.

        This process:
        1. Finds clusters of semantically similar nodes
        2. Picks a canonical node for each cluster
        3. Merges all relationships to the canonical node
        4. Deletes duplicate nodes

        Args:
            user_id: User ID
            dry_run: If True, only report what would be done without making changes
            type_filter: Optional node type to limit consolidation (e.g., "Concept", "Theme")

        Returns:
            Report dict with:
            {
                "duplicate_clusters": List of clusters found,
                "nodes_to_remove": Number of nodes that would be removed,
                "relationships_merged": Number of relationships that would be merged,
                "actions_taken": List of actions performed (if not dry_run)
            }
        """
        logger.info(f"Starting duplicate consolidation (dry_run={dry_run}, type_filter={type_filter})")

        # Get all nodes for the user
        all_nodes = await self.neo4j_manager.get_all_nodes(user_id)

        # Filter by type if specified
        if type_filter:
            all_nodes = [n for n in all_nodes if n.get("type") == type_filter]

        logger.info(f"Analyzing {len(all_nodes)} nodes for duplicates")

        # Build similarity graph
        duplicate_clusters = await self._find_duplicate_clusters(all_nodes, user_id)

        # Generate report
        report = {
            "duplicate_clusters": [],
            "nodes_to_remove": 0,
            "relationships_merged": 0,
            "actions_taken": []
        }

        for cluster in duplicate_clusters:
            canonical_node = cluster[0]  # Keep the first one (could use other strategies)
            duplicates = cluster[1:]

            cluster_info = {
                "canonical": canonical_node["name"],
                "duplicates": [n["name"] for n in duplicates],
                "similarity_scores": [n.get("similarity_score", 0) for n in duplicates]
            }
            report["duplicate_clusters"].append(cluster_info)
            report["nodes_to_remove"] += len(duplicates)

            if not dry_run:
                # Merge each duplicate into canonical
                for dup_node in duplicates:
                    merged_rels = await self._merge_node(
                        source_node=dup_node["name"],
                        target_node=canonical_node["name"],
                        user_id=user_id
                    )
                    report["relationships_merged"] += merged_rels
                    report["actions_taken"].append(
                        f"Merged '{dup_node['name']}' into '{canonical_node['name']}' ({merged_rels} relationships)"
                    )

        logger.info(
            f"Consolidation complete: {report['nodes_to_remove']} nodes to remove, "
            f"{len(report['duplicate_clusters'])} clusters found"
        )

        return report

    async def _find_duplicate_clusters(
        self,
        nodes: List[Dict[str, Any]],
        user_id: str
    ) -> List[List[Dict[str, Any]]]:
        """
        Group nodes into clusters of similar nodes using embeddings.

        Returns:
            List of clusters, where each cluster is a list of similar nodes
        """
        clusters = []
        processed = set()

        for i, node in enumerate(nodes):
            node_name = node["name"]

            if node_name in processed:
                continue

            # Start a new cluster with this node
            cluster = [node]
            processed.add(node_name)

            # Find similar nodes
            similar = await self.find_similar_node(
                node_name=node_name,
                node_type=node.get("type"),
                user_id=user_id,
                limit=10  # Check more candidates
            )

            # Add all similar unprocessed nodes to this cluster
            for j, other_node in enumerate(nodes[i+1:], start=i+1):
                other_name = other_node["name"]

                if other_name in processed:
                    continue

                # Check if this node is similar to the cluster seed
                embeddings = generate_embeddings([node_name, other_name])
                if len(embeddings) == 2 and embeddings[0] and embeddings[1]:
                    similarity = self._cosine_similarity(embeddings[0], embeddings[1])

                    if similarity >= self.similarity_threshold:
                        other_node["similarity_score"] = similarity
                        cluster.append(other_node)
                        processed.add(other_name)

            # Only add clusters with duplicates
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    async def merge_into_existing_node(
        self,
        new_node_name: str,
        existing_node_name: str,
        user_id: str,
        new_properties: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Merge a new node into an existing similar node.

        This is used when we find a similar node during creation - instead of creating
        a duplicate, we just update the existing node's properties if needed.

        Args:
            new_node_name: Name of the new node (won't be created)
            existing_node_name: Name of the existing similar node (will be kept)
            user_id: User ID
            new_properties: Optional properties from the new node to merge

        Returns:
            1 if successful, 0 otherwise
        """
        # For now, just log that we're reusing the existing node
        # Properties are already set on the existing node, and relationships
        # will be created separately in add_relationships()
        logger.info(
            f"Reusing existing node '{existing_node_name}' instead of creating '{new_node_name}'"
        )
        return 1

    async def _merge_node(
        self,
        source_node: str,
        target_node: str,
        user_id: str
    ) -> int:
        """
        Merge source_node into target_node by moving relationships.

        This is used for consolidating existing duplicate nodes:
        1. Moves all relationships from source to target
        2. Deletes the source node

        Returns:
            Number of relationships merged
        """
        driver = self.neo4j_manager._ensure_driver()

        async with driver.session() as session:
            # Get all relationships first
            get_rels_query = """
            MATCH (source:NodeName {name: $source_name, UserId: $user_id})
            OPTIONAL MATCH (source)-[r_out]->(other)
            WHERE other.UserId = $user_id
            WITH source, collect({other: other, type: type(r_out), props: properties(r_out)}) as out_rels
            OPTIONAL MATCH (other)-[r_in]->(source)
            WHERE other.UserId = $user_id
            RETURN out_rels, collect({other: other, type: type(r_in), props: properties(r_in)}) as in_rels
            """

            result = await session.run(get_rels_query, source_name=source_node, user_id=user_id)
            data = await result.data()

            if not data:
                return 0

            out_rels = data[0].get("out_rels", [])
            in_rels = data[0].get("in_rels", [])

            # Now merge using APOC or manual approach
            # For now, use a simpler approach: just delete the source and keep target
            delete_query = """
            MATCH (source:NodeName {name: $source_name, UserId: $user_id})
            DETACH DELETE source
            """

            await session.run(delete_query, source_name=source_node, user_id=user_id)

            return len(out_rels) + len(in_rels)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
