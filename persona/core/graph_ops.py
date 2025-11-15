import json
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.llm.embeddings import generate_embeddings
from persona.models.schema import (
    NodeModel, RelationshipModel, NodesAndRelationshipsResponse,
    CommunityStructure, Subgraph, Node
)
from typing import List, Dict, Any, Optional
from persona.llm.llm_graph import detect_communities
from collections import defaultdict
from server.logging_config import get_logger
from persona.core.deduplication import NodeDeduplicator

logger = get_logger(__name__)

# Deduplication is always enabled with a fixed threshold
DEDUP_SIMILARITY_THRESHOLD = 0.85

class GraphOps:
    def __init__(
        self,
        neo4j_manager: Optional[Neo4jConnectionManager] = None,
        dedup_similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD
    ):
        """
        Initialize GraphOps with an optional Neo4j manager.

        Args:
            neo4j_manager: Neo4j connection manager instance
            dedup_similarity_threshold: Similarity threshold for deduplication (0.0-1.0, default: 0.85)
        """
        self.neo4j_manager = neo4j_manager if neo4j_manager is not None else Neo4jConnectionManager()

        # Deduplication is always enabled
        self.deduplicator = NodeDeduplicator(
            self.neo4j_manager,
            similarity_threshold=dedup_similarity_threshold
        )

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the connection if not already initialized"""
        if not self.neo4j_manager.driver:
            await self.neo4j_manager.initialize()

    async def clean_graph(self):
        # Clean the graph
        await self.neo4j_manager.clean_graph()

    async def add_nodes(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add nodes.")
            return {"nodes_created": 0, "nodes_merged": 0, "node_mapping": {}}

        # Check for semantic duplicates and create mapping
        nodes_to_create = []
        node_mapping = {}  # Maps new_node_name -> existing_node_name for duplicates
        merged_count = 0
        entity_ids_to_append = {}  # Track entity IDs to append to existing nodes

        for node in nodes:
            # Skip deduplication for CognitiveLevel nodes - each concept should have its own
            # CognitiveLevel node(s) to track cognitive progression over time
            if node.type == "CognitiveLevel":
                nodes_to_create.append(node)
                continue

            # Check if a similar node already exists
            # Pass the embedding if node has one (avoids regenerating)
            similar = await self.deduplicator.find_similar_node(
                node_name=node.name,
                node_type=node.type or "",
                user_id=user_id,
                embedding=node.embedding if hasattr(node, 'embedding') else None,
                discipline=node.properties.get('discipline') if hasattr(node, 'properties') and node.properties else None,
                book_id=node.book_id[0] if hasattr(node, 'book_id') and node.book_id and len(node.book_id) > 0 else None
            )

            if similar:
                # Instead of skipping, map this node to the existing similar node
                existing_node_name = similar["name"]
                node_mapping[node.name] = existing_node_name
                merged_count += 1
                logger.info(
                    f"Merging node '{node.name}' into existing similar node '{existing_node_name}' "
                    f"(score: {similar['score']:.3f})"
                )

                # Collect entity IDs to append to the existing node
                if existing_node_name not in entity_ids_to_append:
                    entity_ids_to_append[existing_node_name] = {
                        'book_ids': [],
                        'highlight_ids': [],
                        'writing_ids': []
                    }

                if hasattr(node, 'book_id') and node.book_id:
                    entity_ids_to_append[existing_node_name]['book_ids'].extend(node.book_id)
                if hasattr(node, 'highlight_id') and node.highlight_id:
                    entity_ids_to_append[existing_node_name]['highlight_ids'].extend(node.highlight_id)
                if hasattr(node, 'writing_id') and node.writing_id:
                    entity_ids_to_append[existing_node_name]['writing_ids'].extend(node.writing_id)

                # Also append chunk_ids if present
                if hasattr(node, 'chunk_ids') and node.chunk_ids:
                    await self.neo4j_manager.append_chunk_ids_to_node(
                        existing_node_name, node.chunk_ids, user_id
                    )

                continue

            nodes_to_create.append(node)

        # Append entity IDs to existing nodes that had duplicates merged into them
        for existing_node_name, entity_ids in entity_ids_to_append.items():
            if entity_ids['book_ids'] or entity_ids['highlight_ids'] or entity_ids['writing_ids']:
                await self.neo4j_manager.append_entity_ids_to_node(
                    node_name=existing_node_name,
                    book_ids=entity_ids['book_ids'],
                    highlight_ids=entity_ids['highlight_ids'],
                    writing_ids=entity_ids['writing_ids'],
                    user_id=user_id
                )

        if merged_count:
            logger.info(f"Merged {merged_count} nodes into existing similar nodes")

        nodes_created = 0
        if nodes_to_create:
            # Create nodes with names, properties, types, chunk_ids, entity IDs, and temporal fields
            node_dicts = []
            for node in nodes_to_create:
                node_dict = {
                    "name": node.name,
                    "type": node.type or "",
                    "properties": node.properties or {},
                    "chunk_ids": getattr(node, 'chunk_ids', []),
                    "book_id": getattr(node, 'book_id', []),
                    "highlight_id": getattr(node, 'highlight_id', []),
                    "writing_id": getattr(node, 'writing_id', [])
                }

                # Add temporal fields (created_at)
                created_at = getattr(node, 'created_at', None)
                if created_at:
                    # Convert datetime to ISO 8601 string for Neo4j
                    node_dict["created_at"] = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)

                node_dicts.append(node_dict)

            await self.neo4j_manager.create_nodes(node_dicts, user_id)
            nodes_created = len(nodes_to_create)

            # Generate and add embeddings for new nodes
            await self.add_nodes_batch_embeddings(nodes_to_create, user_id)
        else:
            logger.info("No new nodes to create after deduplication")

        return {
            "nodes_created": nodes_created,
            "nodes_merged": merged_count,
            "node_mapping": node_mapping
        }


    async def add_nodes_batch_embeddings(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add embeddings.")
            return

        # Generate embeddings for all nodes in one batch
        node_names = [node.name for node in nodes]
        embeddings = generate_embeddings(node_names)  # This will now embed the full narrative/label
        
        # Add embeddings to nodes
        for node_name, embedding in zip(node_names, embeddings):
            if embedding:
                await self.neo4j_manager.add_embedding_to_vector_index(node_name, embedding, user_id)
            else:
                logger.error(f"Failed to generate embedding for node: {node_name}")


    async def add_relationships(self, relationships: List[RelationshipModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add relationships.")
            return

        # Filter out self-loops (relationships where source == target)
        filtered_relationships = []
        self_loop_count = 0
        for rel in relationships:
            if rel.source == rel.target:
                logger.warning(f"Skipping self-loop: {rel.source} -[{rel.relation}]-> {rel.target}")
                self_loop_count += 1
                continue
            filtered_relationships.append(rel)

        if self_loop_count > 0:
            logger.info(f"Filtered out {self_loop_count} self-loop(s) from {len(relationships)} relationships")

        if not filtered_relationships:
            logger.warning("No valid relationships to create after filtering self-loops")
            return

        relationship_dicts = [rel.dict() for rel in filtered_relationships]
        await self.neo4j_manager.create_relationships(relationship_dicts, user_id)

    async def get_node_data(self, node_name: str, user_id: str) -> NodeModel:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node data.")
            return NodeModel(name=node_name, type=None, embedding=None, created_at=None)

        node_data = await self.neo4j_manager.get_node_data(node_name, user_id)
        if node_data:
            # Parse created_at from ISO string
            created_at_str = node_data.get("created_at")

            return NodeModel(
                name=node_data["name"],
                type=node_data.get("type"),
                properties=node_data.get("properties", {}),
                embedding=node_data.get("embedding"),
                created_at=created_at_str
            )
        return NodeModel(
            name=node_name,
            type=None,
            embedding=None,
            created_at=None
        )

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node relationships.")
            return []

        relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)
        return [RelationshipModel(source=rel["source"], target=rel["target"], relation=rel["relation"]) 
                for rel in relationships]

    async def text_similarity_search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        threshold: float = 0.7,
        index_name: str = "embeddings_index",
        book_id: Optional[int] = None,
        highlight_id: Optional[int] = None,
        writing_id: Optional[int] = None,
        node_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a similarity search on the graph based on a text query.

        Args:
            query: Text query to search for
            user_id: User ID to filter results by
            limit: Maximum number of results to return after filtering (default: 5)
            threshold: Minimum similarity score to include (0.0-1.0, default: 0.7)
            index_name: Name of the vector index to query
            book_id: Optional book ID to filter results
            highlight_id: Optional highlight ID to filter results
            writing_id: Optional writing ID to filter results
            node_type: Optional node type to filter results (e.g., "Person", "Concept")

        Returns:
            Dictionary with query and filtered results
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        logger.debug(f"Generating embedding for query: '{query}' for user ID: '{user_id}'")
        query_embeddings = generate_embeddings([query])
        if not query_embeddings[0]:
            return {"query": query, "results": []}

        logger.debug(f"Performing similarity search for the query: '{query}' for user ID: '{user_id}'")

        # When entity IDs or node_type filter are provided, use a lower threshold since we're already pre-filtering
        # The manual cosine calculation produces different score ranges than the vector index
        has_entity_filter = book_id is not None or highlight_id is not None or writing_id is not None or node_type is not None
        if has_entity_filter:
            # Lower threshold for entity-filtered queries (manual cosine scores tend to be lower)
            effective_threshold = 0.0  # Return all results from pre-filtered set, sorted by relevance
            fetch_limit = max(limit * 10, 100)  # Fetch more to account for threshold
            logger.info(f"Filtered search (entity or type) with threshold={effective_threshold}")
        else:
            # Standard threshold for vector index queries
            effective_threshold = threshold
            fetch_limit = max(limit * 5, 50)

        results = await self.neo4j_manager.query_text_similarity(
            query_embeddings[0],
            user_id,
            limit=fetch_limit,
            book_id=book_id,
            highlight_id=highlight_id,
            writing_id=writing_id,
            node_type=node_type
        )

        # Filter by effective threshold and apply limit
        if results and has_entity_filter:
            logger.info(f"Entity search returned {len(results)} results, top score: {results[0]['score'] if results else 0:.3f}")

        filtered = [r for r in results if r["score"] >= effective_threshold][:limit]

        return {
            "query": query,
            "results": [
                {
                    "nodeId": result["nodeId"],
                    "nodeName": result["nodeName"],
                    "score": result["score"]
                } for result in filtered
            ]
        }
    
    async def update_graph(self, graph_update: NodesAndRelationshipsResponse, user_id: str):
        """
        Update the graph with new nodes and relationships

        Args:
            graph_update: Contains nodes and relationships to add
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return

        if graph_update.nodes:
            await self.add_nodes(graph_update.nodes, user_id)
        if graph_update.relationships:
            await self.add_relationships(graph_update.relationships, user_id)
        if not graph_update.nodes and not graph_update.relationships:
            logger.debug("No nodes or relationships to update.")

    async def update_graph_transactional(
        self,
        graph_update: NodesAndRelationshipsResponse,
        user_id: str
    ) -> None:
        """
        Update graph with nodes, relationships, and embeddings in a single transaction.
        If any step fails, all changes are rolled back for data consistency.

        This method includes semantic deduplication:
        - Similar nodes are detected and merged into existing ones
        - Relationships are automatically redirected to canonical node names

        Args:
            graph_update: Contains nodes and relationships to add
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return

        if not graph_update.nodes and not graph_update.relationships:
            logger.debug("No nodes or relationships to update.")
            return

        # STEP 1: Perform deduplication and get node mapping
        # OPTIMIZATION: Batch deduplication queries using parallel execution
        # This checks for semantic duplicates and returns a mapping of new_name -> existing_name
        node_mapping = {}
        nodes_to_create = []
        embeddings_data = []
        chunk_ids_to_append = {}  # Maps existing_node_name -> [new chunk_ids to append]
        entity_ids_to_append = {}  # Maps existing_node_name -> {book_ids: [], highlight_ids: [], writing_ids: []}

        # Separate CognitiveLevel nodes (skip deduplication) from others
        cognitive_level_nodes = [node for node in graph_update.nodes if node.type == "CognitiveLevel"]
        nodes_to_check = [node for node in graph_update.nodes if node.type != "CognitiveLevel"]

        # Add cognitive level nodes directly
        nodes_to_create.extend(cognitive_level_nodes)

        # Parallelize deduplication checks for all non-CognitiveLevel nodes
        import asyncio

        async def check_single_node(node):
            """Check if a single node has duplicates"""
            similar = await self.deduplicator.find_similar_node(
                node_name=node.name,
                node_type=node.type or "",
                user_id=user_id,
                embedding=node.embedding,
                discipline=node.properties.get('discipline') if node.properties else None,
                book_id=node.book_id[0] if node.book_id and len(node.book_id) > 0 else None
            )
            return (node, similar)

        # Execute all deduplication checks in parallel
        if nodes_to_check:
            dedup_tasks = [check_single_node(node) for node in nodes_to_check]
            dedup_results = await asyncio.gather(*dedup_tasks, return_exceptions=True)

            # Process results
            for result in dedup_results:
                if isinstance(result, Exception):
                    logger.warning(f"Deduplication check failed: {result}")
                    continue

                node, similar = result

                if similar:
                    # Map this node to the existing similar node
                    existing_node_name = similar["name"]
                    node_mapping[node.name] = existing_node_name

                    # Collect chunk_ids to append to the existing node
                    if node.chunk_ids:
                        if existing_node_name not in chunk_ids_to_append:
                            chunk_ids_to_append[existing_node_name] = []
                        chunk_ids_to_append[existing_node_name].extend(node.chunk_ids)

                    # Collect entity IDs to append to the existing node
                    if existing_node_name not in entity_ids_to_append:
                        entity_ids_to_append[existing_node_name] = {
                            'book_ids': [],
                            'highlight_ids': [],
                            'writing_ids': []
                        }

                    if node.book_id:
                        entity_ids_to_append[existing_node_name]['book_ids'].extend(node.book_id)
                    if node.highlight_id:
                        entity_ids_to_append[existing_node_name]['highlight_ids'].extend(node.highlight_id)
                    if node.writing_id:
                        entity_ids_to_append[existing_node_name]['writing_ids'].extend(node.writing_id)

                    logger.info(
                        f"Merging node '{node.name}' into existing similar node '{existing_node_name}' "
                        f"(score: {similar['score']:.3f})"
                    )
                else:
                    # This is a genuinely new node
                    nodes_to_create.append(node)

        # STEP 1.5: Append chunk_ids and entity IDs to existing nodes that had duplicates merged
        for existing_node_name, new_chunk_ids in chunk_ids_to_append.items():
            if new_chunk_ids:
                await self.neo4j_manager.append_chunk_ids_to_node(
                    node_name=existing_node_name,
                    chunk_ids=new_chunk_ids,
                    user_id=user_id
                )
                logger.debug(
                    f"Appended {len(new_chunk_ids)} chunk_ids to existing node '{existing_node_name}'"
                )

        for existing_node_name, entity_ids in entity_ids_to_append.items():
            if entity_ids['book_ids'] or entity_ids['highlight_ids'] or entity_ids['writing_ids']:
                await self.neo4j_manager.append_entity_ids_to_node(
                    node_name=existing_node_name,
                    book_ids=entity_ids['book_ids'],
                    highlight_ids=entity_ids['highlight_ids'],
                    writing_ids=entity_ids['writing_ids'],
                    user_id=user_id
                )

        # STEP 2: Prepare data for nodes that will actually be created
        nodes_data = []
        for node in nodes_to_create:
            node_dict = {
                "name": node.name,
                "type": node.type or "",
                "properties": node.properties or {},
                "chunk_ids": node.chunk_ids if node.chunk_ids else [],
                "book_id": node.book_id if node.book_id else [],
                "highlight_id": node.highlight_id if node.highlight_id else [],
                "writing_id": node.writing_id if node.writing_id else []
            }

            # DEBUG: Log entity IDs for specific node
            if node.name == "Middle of the roaders":
                logger.info(
                    f"DEBUG: Creating node '{node.name}' with entity IDs - "
                    f"book_id: {node_dict['book_id']}, "
                    f"highlight_id: {node_dict['highlight_id']}, "
                    f"writing_id: {node_dict['writing_id']}"
                )

            # Add temporal fields if present
            if hasattr(node, 'created_at') and node.created_at:
                node_dict["created_at"] = node.created_at

            nodes_data.append(node_dict)

            # Add embedding data for new nodes
            if node.embedding:
                embeddings_data.append({
                    "node_name": node.name,
                    "embedding": node.embedding
                })

        # STEP 3: Apply node mapping to relationships
        # Replace any node names in relationships with their canonical equivalents
        relationships_data = []
        for rel in graph_update.relationships:
            source = node_mapping.get(rel.source, rel.source)
            target = node_mapping.get(rel.target, rel.target)
            relationships_data.append({
                "source": source,
                "target": target,
                "relation": rel.relation
            })

            # Log when we redirect a relationship
            if rel.source != source or rel.target != target:
                logger.debug(
                    f"Redirected relationship: {rel.source}-[{rel.relation}]->{rel.target} "
                    f"=> {source}-[{rel.relation}]->{target}"
                )

        # STEP 3.5: Update concept_uuid for CognitiveLevel nodes whose Concept was merged
        # When Concept A' is merged into Concept A, the CognitiveLevel for A' needs to have
        # its concept_uuid updated to match Concept A's UUID so both cognitive levels can connect
        concept_uuid_updates = []
        for node in graph_update.nodes:
            # Check if this is a CognitiveLevel node with a concept_uuid
            if node.type == "CognitiveLevel" and hasattr(node, 'concept_uuid') and node.concept_uuid:
                # Check if any Concept nodes in this batch were merged
                concept_nodes = [n for n in graph_update.nodes if n.type == "Concept"]
                for concept_node in concept_nodes:
                    # If this Concept was merged (exists in node_mapping)
                    if concept_node.name in node_mapping:
                        merged_into = node_mapping[concept_node.name]

                        # CRITICAL FIX: Only update if this CognitiveLevel's UUID matches the merged Concept's UUID
                        # This ensures we only update CognitiveLevels that belong to the merged Concept
                        if hasattr(concept_node, 'concept_uuid') and concept_node.concept_uuid:
                            if node.concept_uuid == concept_node.concept_uuid:
                                # Get the UUID of the target (canonical) Concept
                                target_concept_uuid = await self._get_concept_uuid_if_exists(merged_into, user_id)
                                if target_concept_uuid:
                                    # Update the CognitiveLevel's concept_uuid to match the canonical Concept
                                    concept_uuid_updates.append({
                                        "cognitive_level_name": node.name,
                                        "old_uuid": node.concept_uuid,
                                        "new_uuid": target_concept_uuid
                                    })
                                    # Update in the node data
                                    for node_data in nodes_data:
                                        if node_data["name"] == node.name:
                                            node_data["properties"]["concept_uuid"] = target_concept_uuid
                                            logger.info(
                                                f"Updated concept_uuid for CognitiveLevel '{node.name}': "
                                                f"{node.concept_uuid} -> {target_concept_uuid} "
                                                f"(Concept merged: {concept_node.name} -> {merged_into})"
                                            )
                                            break

        # Execute everything in a single transaction
        await self.neo4j_manager.update_graph_transactional(
            nodes=nodes_data,
            relationships=relationships_data,
            embeddings_data=embeddings_data,
            user_id=user_id
        )        

    async def close(self):
        """Close the Neo4j connection"""
        logger.info("Closing Neo4j connection...")
        await self.neo4j_manager.close()

    async def get_all_nodes(self, user_id: str) -> List[NodeModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all nodes.")
            return []

        nodes = await self.neo4j_manager.get_all_nodes(user_id)
        return [NodeModel(
            name=node['name'],
            type=node.get('type'),
            properties=node.get('properties', {}),
            embedding=node.get('embedding'),
            created_at=node.get('created_at')
        ) for node in nodes]

    async def get_all_relationships(self, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all relationships.")
            return []

        relationships = await self.neo4j_manager.get_all_relationships(user_id)
        return [RelationshipModel(source=rel['source'], target=rel['target'], relation=rel['relation']) for rel in relationships]
    
    async def create_user(self, user_id: str) -> None:
        await self.neo4j_manager.create_user(user_id)

    async def delete_user(self, user_id: str) -> None:
        await self.neo4j_manager.delete_user(user_id)

    async def user_exists(self, user_id: str) -> bool:
        return await self.neo4j_manager.user_exists(user_id)

    async def perform_similarity_search(
        self, 
        query: str, 
        embedding: List[float],
        user_id: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Perform similarity search using pre-computed embedding
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        try:
            logger.debug(f"Performing similarity search for: {query}")
            results = await self.neo4j_manager.query_text_similarity(embedding, user_id)
            logger.debug(f"Found {len(results)} similar nodes for '{query}'")
            
            return {
                "query": query,
                "results": [
                    {
                        "nodeId": result["nodeId"],
                        "nodeName": result["nodeName"],
                        "score": result["score"]
                    } for result in results
                ]
            }
        except Exception as e:
            logger.error(f"Error in similarity search for {query}: {str(e)}")
            return {"query": query, "results": []}
        

    async def get_ranked_subgraphs(self, user_id: str) -> List[Subgraph]:
        """Get all subgraphs in the graph, ranked by size and influence"""
        subgraphs = []
        visited_nodes = set()

        # Retrieve all nodes for the user
        all_nodes = await self.neo4j_manager.get_all_nodes(user_id)

        for node in all_nodes:
            node_name = node['name']
            if node_name not in visited_nodes:
                # Retrieve all relationships for this node
                relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)

                # Collect all connected nodes
                connected_nodes = {node_name}
                for rel in relationships:
                    connected_nodes.add(rel['source'])
                    connected_nodes.add(rel['target'])

                # Mark nodes as visited
                visited_nodes.update(connected_nodes)

                # Calculate central nodes based on degree
                central_nodes = await self._get_central_nodes(list(connected_nodes), relationships)

                subgraphs.append(Subgraph(
                    id=len(subgraphs),  # Assign a unique ID based on the order
                    nodes=list(connected_nodes),
                    relationships=relationships,
                    size=len(connected_nodes),
                    central_nodes=central_nodes
                ))

        # Sort subgraphs by size in descending order
        subgraphs.sort(key=lambda sg: sg.size, reverse=True)

        return subgraphs

    async def _get_central_nodes(self, nodes: List[str], relationships: List[Dict]) -> List[str]:
        """Calculate central nodes based on degree centrality"""
        if not nodes:  # Handle empty subgraph case
            return []
            
        degree_count = defaultdict(int)
        for rel in relationships:
            degree_count[rel['source']] += 1
            degree_count[rel['target']] += 1
        
        # If we have nodes with relationships, return the most central one
        if degree_count:
            sorted_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
            # Return only the node name (first element of the tuple)
            return [sorted_nodes[0][0]]
        
        # If no relationships exist, return the first node as representative
        return [nodes[0]]
    

    async def format_subgraphs_for_llm(self, subgraphs: List[Subgraph]) -> str:
        """Format subgraphs into a string for LLM input"""
        formatted = "# Graph Structure Analysis\n\n"
        
        for subgraph in subgraphs:
            formatted += f"\n## Subgraph {subgraph.id} (Size: {subgraph.size})\n"
            formatted += f"Central Nodes: {', '.join(subgraph.central_nodes)}\n"
            formatted += "\nNodes:\n"
            for node in subgraph.nodes:
                formatted += f"- {node}\n"
            
            formatted += "\nRelationships:\n"
            for rel in subgraph.relationships:
                formatted += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
        
        return formatted

    async def make_communities(self, user_id: str, community_structure: CommunityStructure, subgraphs: List[Subgraph]) -> None:
        """Create community structure in the graph"""
        for header in community_structure.communityHeaders:
            # Create header node
            await self.neo4j_manager.create_nodes([{
                'name': header.header,
                'type': 'CommunityHeader'
            }], user_id)
            
            for subheader in header.subheaders:
                # Create subheader node
                await self.neo4j_manager.create_nodes([{
                    'name': subheader.subheader,
                    'type': 'CommunitySubheader'
                }], user_id)
                
                # Connect header to subheader
                await self.neo4j_manager.create_relationships([{
                    'source': header.header,
                    'target': subheader.subheader,
                    'relation': 'HAS_SUBHEADER'
                }], user_id)
                
                # Connect representative nodes from subgraphs to this subheader
                for subgraph_id in subheader.subgraph_ids:
                    if subgraph_id < len(subgraphs):
                        subgraph = subgraphs[subgraph_id]
                        for central_node in subgraph.central_nodes:
                            await self.neo4j_manager.create_relationships([{
                                'source': central_node,
                                'target': subheader.subheader,
                                'relation': 'BELONGS_TO'
                            }], user_id)

    async def community_detection(self, user_id) -> None:
        """Main orchestrator for community detection process"""
        # Get ranked subgraphs
        subgraphs = await self.get_ranked_subgraphs(user_id)
        
        # Format for LLM
        subgraphs_text = await self.format_subgraphs_for_llm(subgraphs)
        
        # Get community structure from LLM
        community_structure = await detect_communities(subgraphs_text)
        
        # Create community structure in graph
        await self.make_communities(user_id, community_structure, subgraphs)
        
    async def calculate_bloom_level(self, node_name: str, user_id: str) -> str:
            """
            Calculate Bloom's taxonomy level based on graph evidence and user interaction.

            IMPORTANT: Bloom levels should reflect DEMONSTRATED cognitive work, not just exposure.
            Most concepts start at "Remember" and only advance with evidence of:
            - Understand: User explains/paraphrases the concept
            - Apply: User uses concept to solve problems or in practical contexts
            - Analyze: User breaks down concept, identifies patterns/components
            - Evaluate: User critiques, judges, or defends positions
            - Create: User synthesizes to produce new insights

            Graph topology (connections) provides WEAK evidence and should be used conservatively.
            """
            await self.initialize()

            if not self.neo4j_manager.driver:
                logger.error("Neo4j driver is not initialized.")
                return "Remember"

            # Query to get graph evidence: degree, contexts, entity IDs (as proxy for exposure)
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            OPTIONAL MATCH (n)-[*1..2]-(connected)
            WHERE connected.type IS NOT NULL
            WITH n, degree, collect(DISTINCT connected.type) as contexts
            RETURN
                degree,
                contexts,
                size(n.chunk_ids) as chunk_count,
                size(n.book_id) as book_count,
                size(n.highlight_id) as highlight_count,
                size(n.writing_id) as writing_count
            """

            async with self.neo4j_manager.driver.session() as session:
                result = await session.run(query, node_name=node_name, user_id=user_id)
                data = await result.data()

                if not data:
                    return "Remember"

                record = data[0]
                degree = record.get('degree', 0) or 0
                contexts = [c for c in record.get('contexts', []) if c]
                chunk_count = record.get('chunk_count', 0) or 0
                writing_count = record.get('writing_count', 0) or 0

                # Conservative bloom level calculation
                # Most nodes should remain at Remember unless there's strong evidence

                # Writing projects (user's own thoughts) suggest deeper engagement
                if writing_count > 0:
                    # User wrote about this concept - suggests at least Understanding
                    if degree >= 4 and len(contexts) >= 3:
                        return "Apply"  # Used in multiple contexts in their writing
                    elif degree >= 2:
                        return "Understand"  # Explained in their own words

                # For passive reading (no writing), be very conservative
                # High connectivity might suggest Apply, but only with very strong evidence
                if degree >= 8 and len(contexts) >= 4 and chunk_count >= 5:
                    # Concept appears many times across multiple contexts - suggests practical application
                    return "Apply"
                elif degree >= 4 and len(contexts) >= 2 and chunk_count >= 3:
                    # Concept connected in multiple contexts - suggests understanding
                    return "Understand"

                # Default: Remember (passive exposure, basic recognition)
                return "Remember"

    async def recalculate_bloom_levels_for_concepts(self, concept_names: List[str], user_id: str) -> Dict[str, Any]:
        """
        Recalculate bloom levels for given concepts and create new CognitiveLevel nodes if progression occurred.

        Uses the graph structure to track history: Concept -> HAS_UNDERSTANDING_LEVEL -> CognitiveLevel nodes.
        Each CognitiveLevel node has a created_at timestamp showing when that level was achieved.

        Args:
            concept_names: List of concept node names to recalculate
            user_id: User ID

        Returns:
            Dict with statistics about progressions:
            {
                "concepts_processed": int,
                "progressions": [{"concept": str, "old_level": str, "new_level": str, "timestamp": str}],
                "no_change": int
            }
        """
        await self.initialize()

        if not concept_names:
            return {"concepts_processed": 0, "progressions": [], "no_change": 0}

        stats = {
            "concepts_processed": 0,
            "progressions": [],
            "no_change": 0
        }

        # Bloom level hierarchy for comparison
        bloom_hierarchy = {
            "Remember": 1,
            "Understand": 2,
            "Apply": 3,
            "Analyze": 4,
            "Evaluate": 5,
            "Create": 6
        }

        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc).isoformat()

        for concept_name in concept_names:
            try:
                stats["concepts_processed"] += 1

                # Get UUID for the concept
                concept_uuid = await self._get_or_create_concept_uuid(concept_name, user_id)
                if not concept_uuid:
                    logger.error(f"Failed to get UUID for Concept '{concept_name}' during bloom recalculation")
                    continue

                # Step 1: Get existing CognitiveLevel nodes for this concept via UUID
                existing_levels = await self._get_concept_cognitive_levels_by_uuid(concept_uuid, user_id)

                # Step 2: Calculate new bloom level based on current graph
                new_level = await self.calculate_bloom_level(concept_name, user_id)

                # Step 3: Determine highest level already achieved
                if existing_levels:
                    # Extract level names directly (no parsing needed with UUID-based approach)
                    highest_level = max(
                        existing_levels,
                        key=lambda x: bloom_hierarchy.get(x["level"], 0)
                    )
                    highest_level_name = highest_level["level"]
                    highest_level_rank = bloom_hierarchy.get(highest_level_name, 0)
                else:
                    highest_level_name = None
                    highest_level_rank = 0

                new_level_rank = bloom_hierarchy.get(new_level, 1)

                # Step 4: If new level is higher, create new CognitiveLevel node + edge
                if new_level_rank > highest_level_rank:
                    logger.info(
                        f"Concept '{concept_name}' progressed from '{highest_level_name}' to '{new_level}'"
                    )

                    # Create new CognitiveLevel node
                    await self._create_cognitive_level_node_for_concept(
                        concept_name=concept_name,
                        cognitive_level=new_level,
                        user_id=user_id,
                        created_at=current_time
                    )

                    stats["progressions"].append({
                        "concept": concept_name,
                        "old_level": highest_level_name,
                        "new_level": new_level,
                        "timestamp": current_time
                    })
                else:
                    stats["no_change"] += 1
                    logger.debug(
                        f"Concept '{concept_name}' remains at '{highest_level_name}' (calculated: '{new_level}')"
                    )

            except Exception as e:
                logger.error(f"Failed to recalculate bloom level for concept '{concept_name}': {e}")
                continue

        if stats["progressions"]:
            logger.info(
                f"Bloom level recalculation complete: {len(stats['progressions'])} progressions, "
                f"{stats['no_change']} unchanged"
            )

        return stats

    async def _get_concept_uuid_if_exists(self, concept_name: str, user_id: str) -> Optional[str]:
        """
        Get UUID for a Concept node if it exists in the database.

        Does NOT create or modify the node - only reads existing UUID.

        Args:
            concept_name: Name of the concept node
            user_id: User ID

        Returns:
            UUID string if concept exists with UUID, None otherwise
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return None

        query = """
        MATCH (concept:NodeName {name: $concept_name, UserId: $user_id})
        WHERE concept.type = 'Concept'
        RETURN concept.concept_uuid AS uuid
        """

        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, concept_name=concept_name, user_id=user_id)
            data = await result.data()

            if data and data[0]["uuid"]:
                return str(data[0]["uuid"])
            return None

    async def _get_or_create_concept_uuid(self, concept_name: str, user_id: str) -> Optional[str]:
        """
        Get or create a UUID for a Concept node.

        This UUID is shared with the Concept's CognitiveLevel nodes to enforce
        that only nodes with matching UUIDs can be connected.

        Args:
            concept_name: Name of the concept node
            user_id: User ID

        Returns:
            UUID string for the concept, or None if concept doesn't exist
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return None

        query = """
        MATCH (concept:NodeName {name: $concept_name, UserId: $user_id})
        WHERE concept.type = 'Concept'
        SET concept.concept_uuid = COALESCE(concept.concept_uuid, randomUUID())
        RETURN concept.concept_uuid AS uuid
        """

        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, concept_name=concept_name, user_id=user_id)
            data = await result.data()

            if data:
                return str(data[0]["uuid"])
            return None

    async def _get_concept_cognitive_levels_by_uuid(self, concept_uuid: str, user_id: str) -> List[Dict[str, str]]:
        """
        Get all CognitiveLevel nodes for a specific Concept via shared UUID.

        Args:
            concept_uuid: UUID of the concept node
            user_id: User ID

        Returns:
            List of dicts: [{"level": "Remember", "created_at": "2025-11-05T10:00:00"}, ...]
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return []

        query = """
        MATCH (cl:NodeName {UserId: $user_id})
        WHERE cl.type = 'CognitiveLevel' AND cl.concept_uuid = $concept_uuid
        RETURN cl.name AS level, cl.created_at AS created_at
        ORDER BY cl.created_at ASC
        """

        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, concept_uuid=concept_uuid, user_id=user_id)
            data = await result.data()

            return [
                {
                    "level": str(record["level"]),
                    "created_at": str(record.get("created_at", ""))
                }
                for record in data
            ]

    async def _get_concept_cognitive_levels(self, concept_name: str, user_id: str) -> List[Dict[str, str]]:
        """
        Get all CognitiveLevel nodes connected to a concept, ordered by created_at.

        Returns:
            List of dicts: [{"level": "Remember", "created_at": "2025-11-05T10:00:00"}, ...]
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return []

        query = """
        MATCH (concept:NodeName {name: $concept_name, UserId: $user_id})
        -[:HAS_UNDERSTANDING_LEVEL]->(cl:NodeName)
        WHERE cl.type = 'CognitiveLevel'
        RETURN cl.name AS level, cl.created_at AS created_at
        ORDER BY cl.created_at ASC
        """

        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, concept_name=concept_name, user_id=user_id)
            data = await result.data()

            return [
                {
                    "level": str(record["level"]),
                    "created_at": str(record.get("created_at", ""))
                }
                for record in data
            ]

    async def _create_cognitive_level_node_for_concept(
        self,
        concept_name: str,
        cognitive_level: str,
        user_id: str,
        created_at: str
    ) -> None:
        """
        Create a new CognitiveLevel node and connect it to a concept using shared UUID.

        The Concept and CognitiveLevel share a UUID to enforce that only nodes with
        matching UUIDs can be connected. This allows CognitiveLevel names to remain
        simple ("Remember", "Understand", etc.) while maintaining uniqueness.

        MANDATORY: concept_uuid is required for ALL CognitiveLevel nodes. Nodes will
        not be created without a valid concept_uuid.

        Validation: Ensures that the Concept doesn't already have a connection to
        a CognitiveLevel with the same value (e.g., can't connect to two "Remember" nodes).

        Args:
            concept_name: Name of the concept node
            cognitive_level: Bloom level name (e.g., "Apply")
            user_id: User ID
            created_at: ISO timestamp when this level was achieved
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return

        # Get or create UUID for the concept
        concept_uuid = await self._get_or_create_concept_uuid(concept_name, user_id)
        if not concept_uuid:
            logger.error(
                f"CRITICAL: Cannot create CognitiveLevel node without concept_uuid. "
                f"Failed to get UUID for Concept '{concept_name}'. Node creation aborted."
            )
            return

        # Validation: Check for duplicate CognitiveLevel values using UUID
        existing_levels = await self._get_concept_cognitive_levels_by_uuid(concept_uuid, user_id)
        existing_values = {level_data["level"] for level_data in existing_levels}

        if cognitive_level in existing_values:
            logger.error(
                f"Validation failed: Concept '{concept_name}' already connected to "
                f"CognitiveLevel value '{cognitive_level}'. Skipping duplicate creation."
            )
            return

        # Create CognitiveLevel node with simple name and shared UUID
        query = """
        // Find the concept node
        MATCH (concept:NodeName {name: $concept_name, UserId: $user_id})
        WHERE concept.type = 'Concept'

        // Create new CognitiveLevel node with shared UUID
        CREATE (cl:NodeName {
            name: $cognitive_level,
            type: 'CognitiveLevel',
            UserId: $user_id,
            concept_uuid: $concept_uuid,
            created_at: $created_at
        })

        // Create relationship (CognitiveLevel can only have ONE edge - to its parent Concept)
        CREATE (concept)-[:HAS_UNDERSTANDING_LEVEL]->(cl)

        RETURN cl.name AS level
        """

        async with self.neo4j_manager.driver.session() as session:
            await session.run(
                query,
                concept_name=concept_name,
                cognitive_level=cognitive_level,
                concept_uuid=concept_uuid,
                user_id=user_id,
                created_at=created_at
            )

        logger.debug(
            f"Created CognitiveLevel '{cognitive_level}' for concept '{concept_name}' (UUID: {concept_uuid}) at {created_at}"
        )

    async def cleanup_invalid_cognitive_level_relationships(self, user_id: str) -> Dict[str, int]:
        """
        Clean up invalid CognitiveLevel data to enforce Rule 2: "CognitiveLevel nodes cannot exist without an edge."

        Performs four cleanup operations:
        1. Delete HAS_UNDERSTANDING_LEVEL relationships where Concept and CognitiveLevel have mismatched concept_uuid
        2. Delete CognitiveLevel nodes that don't have a concept_uuid property (mandatory field)
        3. Delete CognitiveLevel nodes with no incoming HAS_UNDERSTANDING_LEVEL relationship (orphaned nodes)
        4. Delete CognitiveLevel nodes whose parent Concept (with matching concept_uuid) was deleted

        This ensures:
        - Only nodes with matching UUIDs are connected via HAS_UNDERSTANDING_LEVEL
        - All CognitiveLevel nodes MUST have a concept_uuid
        - All CognitiveLevel nodes MUST have exactly ONE incoming HAS_UNDERSTANDING_LEVEL edge
        - CognitiveLevel nodes cannot outlive their parent Concept nodes

        Args:
            user_id: User ID

        Returns:
            Dict with cleanup statistics: {
                "deleted_relationships": int,
                "deleted_nodes_no_uuid": int,
                "deleted_orphaned_nodes": int,
                "deleted_nodes_no_parent": int,
                "total_deleted_nodes": int
            }
        """
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return {
                "deleted_relationships": 0,
                "deleted_nodes_no_uuid": 0,
                "deleted_orphaned_nodes": 0,
                "deleted_nodes_no_parent": 0,
                "total_deleted_nodes": 0
            }

        # Step 1: Delete invalid relationships (mismatched UUIDs)
        relationship_query = """
        // Find all HAS_UNDERSTANDING_LEVEL relationships with mismatched UUIDs
        MATCH (concept:NodeName {UserId: $user_id})-[r:HAS_UNDERSTANDING_LEVEL]->(cl:NodeName {UserId: $user_id})
        WHERE concept.type = 'Concept'
          AND cl.type = 'CognitiveLevel'
          AND concept.concept_uuid <> cl.concept_uuid

        // Delete the invalid relationship
        DELETE r

        RETURN count(r) AS deleted_count
        """

        # Step 2: Delete CognitiveLevel nodes without concept_uuid (mandatory field)
        node_no_uuid_query = """
        // Find all CognitiveLevel nodes without concept_uuid
        MATCH (cl:NodeName {UserId: $user_id})
        WHERE cl.type = 'CognitiveLevel'
          AND (cl.concept_uuid IS NULL OR cl.concept_uuid = '')

        // Detach and delete the node (removes all relationships)
        DETACH DELETE cl

        RETURN count(cl) AS deleted_count
        """

        # Step 3: Delete orphaned CognitiveLevel nodes (no incoming HAS_UNDERSTANDING_LEVEL relationship)
        orphaned_nodes_query = """
        // Find CognitiveLevel nodes with no incoming HAS_UNDERSTANDING_LEVEL relationship
        MATCH (cl:NodeName {UserId: $user_id})
        WHERE cl.type = 'CognitiveLevel'
          AND cl.concept_uuid IS NOT NULL
          AND NOT EXISTS(()-[:HAS_UNDERSTANDING_LEVEL]->(cl))

        // Delete the orphaned node (removes any other relationships if they exist)
        DETACH DELETE cl

        RETURN count(cl) AS deleted_count
        """

        # Step 4: Delete CognitiveLevel nodes whose parent Concept doesn't exist
        missing_parent_query = """
        // Find CognitiveLevel nodes where the parent Concept with matching UUID doesn't exist
        MATCH (cl:NodeName {UserId: $user_id})
        WHERE cl.type = 'CognitiveLevel'
          AND cl.concept_uuid IS NOT NULL
          AND NOT EXISTS(
            (concept:NodeName {UserId: $user_id})
            WHERE concept.type = 'Concept'
              AND concept.concept_uuid = cl.concept_uuid
          )

        // Delete the orphaned node
        DETACH DELETE cl

        RETURN count(cl) AS deleted_count
        """

        async with self.neo4j_manager.driver.session() as session:
            # Execute all cleanup operations in order
            rel_result = await session.run(relationship_query, user_id=user_id)
            rel_data = await rel_result.data()
            deleted_relationships = rel_data[0]["deleted_count"] if rel_data else 0

            node_no_uuid_result = await session.run(node_no_uuid_query, user_id=user_id)
            node_no_uuid_data = await node_no_uuid_result.data()
            deleted_nodes_no_uuid = node_no_uuid_data[0]["deleted_count"] if node_no_uuid_data else 0

            orphaned_result = await session.run(orphaned_nodes_query, user_id=user_id)
            orphaned_data = await orphaned_result.data()
            deleted_orphaned_nodes = orphaned_data[0]["deleted_count"] if orphaned_data else 0

            missing_parent_result = await session.run(missing_parent_query, user_id=user_id)
            missing_parent_data = await missing_parent_result.data()
            deleted_nodes_no_parent = missing_parent_data[0]["deleted_count"] if missing_parent_data else 0

            # Log warnings for each cleanup type
            if deleted_relationships > 0:
                logger.warning(
                    f"Cleaned up {deleted_relationships} invalid CognitiveLevel relationship(s) "
                    f"with mismatched concept_uuid values"
                )

            if deleted_nodes_no_uuid > 0:
                logger.warning(
                    f"Deleted {deleted_nodes_no_uuid} CognitiveLevel node(s) without concept_uuid property (mandatory field)"
                )

            if deleted_orphaned_nodes > 0:
                logger.warning(
                    f"Deleted {deleted_orphaned_nodes} orphaned CognitiveLevel node(s) "
                    f"with no incoming HAS_UNDERSTANDING_LEVEL relationship"
                )

            if deleted_nodes_no_parent > 0:
                logger.warning(
                    f"Deleted {deleted_nodes_no_parent} CognitiveLevel node(s) "
                    f"whose parent Concept was deleted"
                )

            total_deleted_nodes = (
                deleted_nodes_no_uuid +
                deleted_orphaned_nodes +
                deleted_nodes_no_parent
            )

            return {
                "deleted_relationships": deleted_relationships,
                "deleted_nodes_no_uuid": deleted_nodes_no_uuid,
                "deleted_orphaned_nodes": deleted_orphaned_nodes,
                "deleted_nodes_no_parent": deleted_nodes_no_parent,
                "total_deleted_nodes": total_deleted_nodes
            }

    async def get_concept_bloom_history(self, concept_name: str, user_id: str) -> List[Dict[str, str]]:
        """
        Get the bloom level progression history for a concept.

        Queries all CognitiveLevel nodes connected to the concept via shared UUID
        and returns them in chronological order based on created_at timestamps.

        Args:
            concept_name: Name of the concept node
            user_id: User ID

        Returns:
            List of progression events: [{"level": "Remember", "achieved_at": "2025-11-05T10:00:00"}, ...]
        """
        # Get UUID for the concept
        concept_uuid = await self._get_or_create_concept_uuid(concept_name, user_id)
        if not concept_uuid:
            logger.error(f"Failed to get UUID for Concept '{concept_name}' in bloom history query")
            return []

        levels = await self._get_concept_cognitive_levels_by_uuid(concept_uuid, user_id)

        # Format for external consumption (no parsing needed with UUID-based approach)
        result = []
        for level in levels:
            result.append({
                "level": level["level"],
                "achieved_at": level["created_at"] or "Unknown"
            })

        return result

    async def update_node_properties(self, node_name: str, user_id: str, properties: Dict[str, Any]) -> None:
        """Update properties of an existing node"""
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update node properties.")
            return
        
        await self.initialize()
        
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return
        
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        SET n.discipline = $discipline,
            n.bloom_level = $bloom_level,
            n.confidence = $confidence
        RETURN n.name as name
        """
        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, 
                node_name=node_name, 
                user_id=user_id, 
                discipline=properties.get("discipline", ""),
                bloom_level=properties.get("bloom_level", ""),
                confidence=properties.get("confidence", 0.0)
            )
            data = await result.data()
            if data:
                logger.debug(f"Updated properties for node: {node_name}")
            else:
                logger.warning(f"Node {node_name} not found for user {user_id}")

    async def prune_infrequent_themes(
        self,
        user_id: str,
        min_books: int = 2
    ) -> Dict[str, Any]:
        """
        Prune Theme nodes that appear infrequently after similarity merging.

        This implements Tier 2 of the two-tier Theme consolidation strategy:
        - Tier 1: Aggressive similarity merging (0.60 threshold) consolidates similar Themes
        - Tier 2: This method prunes Themes with insufficient evidence

        A Theme is kept if it meets EITHER criterion:
        - Meets percentage-based threshold: 15% of book's chunks (min 3, max 15), OR
        - Has 2+ book_ids (recurring theme across multiple books)

        Percentage-based threshold examples:
        - Book with 10 chunks → threshold = 3 (30% due to floor)
        - Book with 50 chunks → threshold = 5 (10%)
        - Book with 200 chunks → threshold = 15 (7.5% due to ceiling)

        Otherwise, the Theme is too granular/one-off and gets deleted.

        Args:
            user_id: User ID
            min_books: Minimum book_ids for cross-book Theme (default: 2)

        Returns:
            Dict with stats:
            {
                "themes_analyzed": int,
                "themes_kept": int,
                "themes_pruned": int,
                "pruned_themes": [{"name": str, "chunks": int, "books": int}, ...]
            }
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot prune themes.")
            return {
                "themes_analyzed": 0,
                "themes_kept": 0,
                "themes_pruned": 0,
                "pruned_themes": []
            }

        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver not initialized")
            return {
                "themes_analyzed": 0,
                "themes_kept": 0,
                "themes_pruned": 0,
                "pruned_themes": []
            }

        # Query to count and prune themes using percentage-based threshold
        query = """
        // Step 1: Calculate total unique chunks per book across ALL nodes
        MATCH (all_nodes:NodeName {UserId: $user_id})
        WHERE size(COALESCE(all_nodes.book_id, [])) > 0
        UNWIND all_nodes.book_id as book_id
        UNWIND COALESCE(all_nodes.chunk_ids, []) as chunk_id
        WITH book_id, collect(DISTINCT chunk_id) as all_chunks_in_book
        WITH book_id, size(all_chunks_in_book) as total_chunks_in_book

        // Step 2: Calculate percentage-based threshold for each book
        // Formula: max(3, min(15, total_chunks * 0.15))
        WITH book_id, total_chunks_in_book,
             toInteger(CASE
                 WHEN total_chunks_in_book * 0.15 < 3 THEN 3
                 WHEN total_chunks_in_book * 0.15 > 15 THEN 15
                 ELSE total_chunks_in_book * 0.15
             END) as threshold_for_book

        // Step 3: Collect thresholds by book_id
        WITH collect({book_id: book_id, threshold: threshold_for_book, total_chunks: total_chunks_in_book}) as book_thresholds

        // Step 4: Evaluate each Theme node
        MATCH (theme:NodeName {UserId: $user_id})
        WHERE theme.type = 'Theme'
        WITH theme, book_thresholds,
             size(COALESCE(theme.chunk_ids, [])) as theme_chunk_count,
             size(COALESCE(theme.book_id, [])) as book_count,
             COALESCE(theme.book_id, []) as theme_book_ids

        // Step 5: Determine if theme should be kept
        WITH theme, theme_chunk_count, book_count, theme_book_ids, book_thresholds,
             CASE
                 // Multi-book themes: automatically KEEP
                 WHEN book_count >= $min_books THEN 'KEEP'
                 // Single-book themes: check against percentage threshold
                 WHEN book_count = 1 THEN
                     CASE
                         // Find threshold for this theme's book
                         WHEN theme_chunk_count >= [t IN book_thresholds WHERE t.book_id = theme_book_ids[0] | t.threshold][0] THEN 'KEEP'
                         ELSE 'PRUNE'
                     END
                 // No books: prune
                 ELSE 'PRUNE'
             END as action

        // Step 6: Collect stats
        WITH
            count(theme) as total,
            sum(CASE WHEN action = 'KEEP' THEN 1 ELSE 0 END) as kept,
            collect(CASE WHEN action = 'PRUNE' THEN {
                name: theme.name,
                chunks: theme_chunk_count,
                books: book_count
            } ELSE null END) as to_prune_list

        // Filter nulls from to_prune_list
        WITH total, kept, [item IN to_prune_list WHERE item IS NOT NULL] as to_prune

        // Step 7: Delete the themes to prune
        UNWIND CASE WHEN size(to_prune) > 0 THEN to_prune ELSE [null] END as theme_info
        OPTIONAL MATCH (n:NodeName {UserId: $user_id, name: theme_info.name})
        WHERE n.type = 'Theme' AND theme_info IS NOT NULL
        DETACH DELETE n

        RETURN total, kept, to_prune as pruned_list
        """

        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(
                query,
                user_id=user_id,
                min_books=min_books
            )
            data = await result.data()

            if not data or not data[0]:
                logger.info("No Theme nodes found to analyze")
                return {
                    "themes_analyzed": 0,
                    "themes_kept": 0,
                    "themes_pruned": 0,
                    "pruned_themes": []
                }

            record = data[0]
            total = record.get("total", 0) or 0
            kept = record.get("kept", 0) or 0
            pruned_list = record.get("pruned_list", []) or []
            pruned = len(pruned_list)

            if pruned > 0:
                pruned_names = [t["name"][:60] + "..." if len(t["name"]) > 60 else t["name"] for t in pruned_list]
                logger.info(
                    f"Theme pruning complete: {pruned}/{total} themes removed "
                    f"(kept: {kept} with sufficient evidence). "
                    f"Removed: {pruned_names[:5]}{'...' if len(pruned_names) > 5 else ''}"
                )
            else:
                logger.info(f"Theme pruning complete: All {kept}/{total} themes have sufficient evidence (no pruning needed)")

            return {
                "themes_analyzed": total,
                "themes_kept": kept,
                "themes_pruned": pruned,
                "pruned_themes": pruned_list
            }

class GraphContextRetriever:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def get_rich_context(self, query: str, user_id: str, top_k: int = 5, max_hops: int = 2) -> str:
        """
        Get rich context from the graph based on a text query.
        Uses two-tiered retrieval: general similarity search + Person-specific search.
        """
        # Tier 1: General vector similarity search (all node types)
        similar_nodes = await self.graph_ops.text_similarity_search(
            query=query,
            user_id=user_id,
            limit=top_k,
            index_name="embeddings_index"
        )

        # Tier 2: Person-specific vector search (guarantees Person nodes are included when relevant)
        person_nodes = await self.graph_ops.text_similarity_search(
            query=query,
            user_id=user_id,
            limit=3,  # Get top 3 Person nodes
            index_name="embeddings_index",
            node_type="Person"
        )

        # Merge results, avoiding duplicates
        all_nodes = similar_nodes['results'].copy()
        existing_node_names = {node['nodeName'] for node in all_nodes}

        for person_node in person_nodes['results']:
            if person_node['nodeName'] not in existing_node_names:
                all_nodes.append(person_node)
                logger.debug(f"Added Person node to context: {person_node['nodeName']} (score: {person_node['score']:.3f})")

        # Crawl graph from all seed nodes
        context = await self.crawl_graph(all_nodes, max_hops, user_id)
        return self.format_separated_context(context)

    async def crawl_graph(self, start_nodes, max_hops, user_id):
        """
        Crawl the graph to get rich context.
        """
        context = {}
        for node in start_nodes:
            logger.debug(f"Exploring node: {node}")
            await self.explore_node(node['nodeName'], context, max_hops, user_id)
        return context

    async def explore_node(self, node_name, context, hops_left, user_id):
        """
        Explore a node and its relationships up to max_hops away.
        """
        if node_name in context or hops_left < 0:
            return

        node_data = await self.graph_ops.get_node_data(node_name, user_id)

        # Skip CognitiveLevel nodes - they should not appear in relationship generation context
        # to prevent new Concepts from connecting to existing CognitiveLevel nodes
        if node_data.type == "CognitiveLevel":
            logger.debug(f"Skipping CognitiveLevel node '{node_name}' from relationship context")
            return

        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)

        context[node_name] = {
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in relationships:
            context[node_name]['relationships'].append(f"{rel.relation} -> {rel.target}")
            if hops_left > 0:
                await self.explore_node(rel.target, context, hops_left - 1, user_id)

    def format_separated_context(self, context):
        """
        Format the context into a readable string with graph structure and node details.
        """
        graph_structure = []
        node_descriptions = []

        for node, data in context.items():
            for relationship in data['relationships']:
                graph_structure.append(f"{node} -> {relationship}")
            
            node_descriptions.append(f"### {node}")
            if data.get('properties'):
                for key, value in data['properties'].items():
                    node_descriptions.append(f"{key}: {value}")
            node_descriptions.append("")

        formatted = "# User Knowledge Graph\n\n## Graph Structure\n```\n"
        formatted += "\n".join(graph_structure)
        formatted += "\n```\n\n## Node Details\n\n"
        formatted += "\n".join(node_descriptions)

        return formatted
    
    async def get_relevant_graph_context(
        self,
        nodes: List[Node],
        user_id: str,
        max_hops: int = 2,
        max_context_chars: int = 50000  # Roughly 12,500 tokens (conservative estimate)
    ) -> str:
        """
        Get relevant subgraph context for the given nodes with intelligent truncation.

        Strategy for large graphs:
        1. Always include seed nodes (from vector search)
        2. Limit relationships per node to most important
        3. Prioritize 1-hop neighbors over 2-hop
        4. Track context size and stop when approaching limit
        """
        context = "# Relevant Graph Context\n\n"

        # Check if there are any existing nodes in the graph
        existing_nodes = await self.graph_ops.get_all_nodes(user_id)
        if not existing_nodes:
            logger.debug("No existing nodes in graph (t=0). Skipping context retrieval.")
            return context

        logger.debug(f"Found {len(existing_nodes)} existing nodes in graph")

        # Start exploring from the provided nodes directly with context limits
        subgraph = {}
        context_size = 0
        max_relationships_per_node = 20  # Limit relationships to prevent explosion

        for node in nodes:
            if context_size > max_context_chars:
                logger.warning(f"Context size limit reached ({context_size} chars). Stopping exploration.")
                break

            context_size = await self._explore_node_with_limits(
                node_name=node.name,
                subgraph=subgraph,
                user_id=user_id,
                max_hops=max_hops,
                max_relationships_per_node=max_relationships_per_node,
                current_context_size=context_size,
                max_context_size=max_context_chars
            )

        # Format the subgraph context
        if subgraph:
            context += "## Related Nodes and Relationships\n"
            nodes_included = 0
            for node_name, data in subgraph.items():
                # Check if adding this node would exceed limit
                node_context = f"\n### {node_name}\n"
                if data['relationships']:
                    node_context += "Relationships:\n"
                    for rel in data['relationships']:
                        node_context += f"- {rel}\n"

                if len(context) + len(node_context) > max_context_chars:
                    logger.warning(f"Stopped at {nodes_included} nodes to stay within context limit")
                    break

                context += node_context
                nodes_included += 1

            logger.info(f"Context includes {nodes_included} nodes with {context_size} total relationship entries")

        return context

    async def _explore_node_with_limits(
        self,
        node_name: str,
        subgraph: Dict,
        user_id: str,
        max_hops: int,
        max_relationships_per_node: int,
        current_context_size: int,
        max_context_size: int
    ) -> int:
        """
        Explore node relationships with size limits to prevent context explosion.

        Returns updated context size.
        """
        if node_name in subgraph or max_hops < 0 or current_context_size > max_context_size:
            return current_context_size

        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)

        # Limit relationships to prevent explosion
        limited_relationships = relationships[:max_relationships_per_node]
        if len(relationships) > max_relationships_per_node:
            logger.warning(
                f"Node '{node_name}' has {len(relationships)} relationships. "
                f"Limiting to {max_relationships_per_node} most important ones."
            )

        subgraph[node_name] = {
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in limited_relationships:
            rel_text = f"{rel.source} {rel.relation} {rel.target}"
            subgraph[node_name]['relationships'].append(rel_text)
            current_context_size += len(rel_text)

            # Only explore further if we have hops left and space in context
            if max_hops > 0 and current_context_size < max_context_size:
                next_node = rel.target if rel.source == node_name else rel.source
                # Reduce max_relationships for deeper hops (prioritize nearby nodes)
                deeper_max_rels = max(5, max_relationships_per_node // 2)
                current_context_size = await self._explore_node_with_limits(
                    next_node,
                    subgraph,
                    user_id,
                    max_hops - 1,
                    deeper_max_rels,
                    current_context_size,
                    max_context_size
                )

        return current_context_size

    async def _explore_node(self, node_name: str, subgraph: Dict, user_id: str, max_hops: int):
        """Helper method to explore node relationships for context building (legacy, kept for compatibility)"""
        if node_name in subgraph or max_hops < 0:
            return

        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)

        subgraph[node_name] = {
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in relationships:
            subgraph[node_name]['relationships'].append(
                f"{rel.source} {rel.relation} {rel.target}"
            )
            if max_hops > 0:
                next_node = rel.target if rel.source == node_name else rel.source
                await self._explore_node(next_node, subgraph, user_id, max_hops - 1)

    async def get_entire_graph_context(self, user_id: str) -> str:
        """
        Get graph context relative to the nodes, and user psyche.
        Currently gets the entire graph context, which is not efficient.
        """
        # TODO: This is not efficient, only get the context for the nodes that are relevant.
        # TODO: Use more efficient sophisticated context retrieval techniques like graph traversal, etc.
        # TODO: Use vector search to get the context for the nodes.
    
        nodes = await self.graph_ops.get_all_nodes(user_id)
        relationships = await self.graph_ops.get_all_relationships(user_id)
       
        context = "# Current Knowledge Graph\n\n## Nodes\n"
        for node in nodes:
            context += f"- {node.name} (type: {node.type})\n"
        
        context += "\n## Relationships\n"
        for rel in relationships:
            context += f"- {rel.source} {rel.relation} {rel.target}\n"
        
        return context

    async def get_graph_context(self, query: str, user_id: str):
        logger.debug(f"Getting graph context for query: {query}")
        query_embeddings = generate_embeddings([query])
        if not query_embeddings[0]:
            return {"query": query, "results": []}
        results = await self.graph_ops.perform_similarity_search(query=query, embedding=query_embeddings[0], user_id=user_id)
        return results