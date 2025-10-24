from typing import List, Dict, Any, Union, Tuple, Optional
from neo4j import AsyncGraphDatabase, basic_auth, AsyncDriver
import asyncio
import time
from persona.llm.embeddings import generate_embeddings
import json
from server.config import config
from server.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jConnectionManager:
    def __init__(self):
        self.uri = config.NEO4J.URI
        self.username = config.NEO4J.USER
        self.password = config.NEO4J.PASSWORD
        self.driver: Optional[AsyncDriver] = None
        self.ensure_vector_index_task = None

    async def initialize(self):
        """Initialize the connection and wait for Neo4j to be ready"""
        await self.connect()
        await self.wait_for_neo4j()
        await self.ensure_vector_index()

    async def connect(self):
        """Create the driver connection"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=basic_auth(self.username, self.password),
            max_connection_lifetime=3600
        )

    async def wait_for_neo4j(self, timeout=60):
        """Wait for Neo4j to be ready"""
        start_time = time.time()
        while True:
            try:
                if not self.driver:
                    await self.connect()
                if self.driver:
                    async with self._ensure_driver().session() as session:
                        await session.run("RETURN 1")
                        logger.info("Neo4j is ready.")
                        return
            except Exception as e:
                logger.debug(f"Waiting for Neo4j... {str(e)}")
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    logger.error(f"Failed to connect to Neo4j after {timeout} seconds.")
                    raise e
                await asyncio.sleep(2)  # Increased sleep time

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None

    def _ensure_driver(self) -> AsyncDriver:
        """Ensure driver is initialized and return it"""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call initialize() first.")
        return self.driver

    async def clean_graph(self) -> None:
        # Delete all nodes and relationships
        async with self._ensure_driver().session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        # Drop the vector index
        await self.drop_vector_index("embeddings_index")

    async def check_node_exists(self, node_name: str, node_type: str, user_id: str) -> bool:
        query = """
        MATCH (n {name: $node_name, NodeType: $node_type, UserId: $user_id})
        RETURN n.name AS NodeName
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, node_name=node_name, node_type=node_type, user_id=user_id)
            return result.single() is not None

    async def drop_vector_index(self, index_name: str) -> None:
        if await self.index_exists(index_name):
            query = "DROP INDEX $index_name"
            async with self._ensure_driver().session() as session:
                await session.run(query, index_name=index_name)
            logger.info(f"Vector index '{index_name}' dropped.")
        else:
            logger.debug(f"Vector index '{index_name}' does not exist. Skipping drop operation.")

    async def create_nodes(self, nodes: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot create nodes.")
            return
        async with self._ensure_driver().session() as session:
            for node in nodes:
                # Store PKG properties as individual fields, not nested JSON
                # Use node type as additional label for better visualization
                node_type = (node.get("type") or "Unknown").replace(" ", "")
                properties = node.get("properties", {})

                # Base query with standard fields including temporal fields
                query = (
                    f"MERGE (n:NodeName:`{node_type}` {{name: $name, UserId: $user_id}}) "
                    "SET n.type = $type, "
                    "n.discipline = $discipline, "
                    "n.bloom_level = $bloom_level, "
                    "n.confidence = $confidence"
                )

                params = {
                    "name": node["name"],
                    "user_id": user_id,
                    "type": node.get("type", ""),
                    "discipline": properties.get("discipline", ""),
                    "bloom_level": properties.get("bloom_level", ""),
                    "confidence": properties.get("confidence", 0.0)
                }

                # Handle temporal fields (created_at and bloom_history)
                # Only set created_at if it doesn't exist (preserve original timestamp)
                created_at = node.get("created_at")
                if created_at:
                    query += ", n.created_at = COALESCE(n.created_at, $created_at)"
                    params["created_at"] = created_at

                # Set bloom_history if provided
                # Neo4j doesn't support arrays of maps, so store as JSON string
                bloom_history = node.get("bloom_history")
                if bloom_history:
                    query += ", n.bloom_history = $bloom_history"
                    params["bloom_history"] = json.dumps(bloom_history) if bloom_history else None

                # Handle chunk_ids array
                chunk_ids = node.get("chunk_ids", [])
                if chunk_ids:
                    query += ", n.chunk_ids = $chunk_ids"
                    params["chunk_ids"] = chunk_ids

                # Handle entity ID arrays (book_id, highlight_id, writing_id)
                # Always set these fields, even if empty, to ensure they exist in the database
                for id_field in ['book_id', 'highlight_id', 'writing_id']:
                    ids = node.get(id_field, [])
                    query += f", n.{id_field} = ${id_field}"
                    params[id_field] = ids

                # Add custom properties dynamically (exclude standard PKG properties and entity IDs)
                if properties:
                    standard_props = {"discipline", "bloom_level", "confidence", "type", "chunk_ids", "book_id", "highlight_id", "writing_id"}
                    custom_props = {k: v for k, v in properties.items() if k not in standard_props}

                    logger.debug(f"Node {node['name']}: Found {len(custom_props)} custom properties: {list(custom_props.keys())}")

                    for prop_key, prop_value in custom_props.items():
                        # Sanitize property key to be Cypher-safe
                        safe_key = prop_key.replace(" ", "_").replace("-", "_")
                        query += f", n.{safe_key} = ${safe_key}"
                        params[safe_key] = str(prop_value)

                logger.debug(f"Final query for node {node['name']}: {query}")
                logger.debug(f"Params: {params}")
                await session.run(query, params)  # type: ignore

    async def create_relationships(self, relationships: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot create relationships.")
            return
        async with self._ensure_driver().session() as session:
            for relationship in relationships:
                await self._create_relationship(session, relationship, user_id)

    async def _create_relationship(self, session, relationship: Dict[str, Any], user_id: str) -> None:
        # Base query to create relationship
        query = (
            "MATCH (source {UserId: $user_id}), (target {UserId: $user_id}) "
            "WHERE source.name = $source AND target.name = $target "
            "MERGE (source)-[r:`{relation}`]->(target) "
            "SET r.value = $relation"
        )

        params = {
            "source": relationship["source"],
            "target": relationship["target"],
            "relation": relationship["relation"],
            "user_id": user_id
        }

        # Add properties if they exist
        properties = relationship.get("properties", {})
        if properties:
            for prop_key, prop_value in properties.items():
                # Sanitize property key
                safe_key = prop_key.replace(" ", "_").replace("-", "_")
                query += f", r.{safe_key} = ${safe_key}"
                params[safe_key] = str(prop_value)

        await session.run(query, params)

    async def update_graph_transactional(
        self,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        embeddings_data: List[Dict[str, Any]],
        bloom_updates: List[Dict[str, Any]],
        user_id: str
    ) -> None:
        """
        Execute entire graph update in a single transaction for atomicity.
        If any step fails, all changes are rolled back.
        
        Args:
            nodes: List of node dicts with 'name', 'type', 'properties'
            relationships: List of relationship dicts with 'source', 'target', 'relation'
            embeddings_data: List of dicts with 'node_name' and 'embedding'
            bloom_updates: List of dicts with 'node_name' and 'properties' (including bloom_level)
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return
        
        async with self._ensure_driver().session() as session:
            tx = await session.begin_transaction()
            try:
                # Step 1: Create/update nodes
                for node in nodes:
                    # Store PKG properties as individual fields
                    # Use node type as additional label for better visualization
                    node_type = (node.get("type") or "Unknown").replace(" ", "")
                    query = (
                        f"MERGE (n:NodeName:`{node_type}` {{name: $name, UserId: $user_id}}) "
                        "SET n.type = $type, "
                        "n.discipline = $discipline, "
                        "n.bloom_level = $bloom_level, "
                        "n.confidence = $confidence"
                    )
                    properties = node.get("properties", {})
                    params = {
                        "name": node["name"],
                        "user_id": user_id,
                        "type": node.get("type", ""),
                        "discipline": properties.get("discipline", ""),
                        "bloom_level": properties.get("bloom_level", ""),
                        "confidence": properties.get("confidence", 0.0)
                    }

                    # Handle temporal fields (created_at and bloom_history)
                    # Only set created_at if it doesn't exist (preserve original timestamp)
                    created_at = node.get("created_at")
                    if created_at:
                        query += ", n.created_at = COALESCE(n.created_at, $created_at)"
                        params["created_at"] = created_at

                    # Set bloom_history if provided
                    # Neo4j doesn't support arrays of maps, so store as JSON string
                    bloom_history = node.get("bloom_history")
                    if bloom_history:
                        query += ", n.bloom_history = $bloom_history"
                        params["bloom_history"] = json.dumps(bloom_history) if bloom_history else None

                    # Handle chunk_ids array
                    chunk_ids = node.get("chunk_ids", [])
                    if chunk_ids:
                        query += ", n.chunk_ids = $chunk_ids"
                        params["chunk_ids"] = chunk_ids

                    # Handle entity ID arrays (book_id, highlight_id, writing_id)
                    # Always set these fields, even if empty, to ensure they exist in the database
                    for id_field in ['book_id', 'highlight_id', 'writing_id']:
                        ids = node.get(id_field, [])
                        query += f", n.{id_field} = ${id_field}"
                        params[id_field] = ids

                    await tx.run(query, params) # type: ignore
                    logger.debug(f"Transaction: Created/updated node {node['name']}")
                # Step 2: Create relationships
                for relationship in relationships:
                    query = (
                        "MATCH (source {UserId: $user_id}), (target {UserId: $user_id}) "
                        "WHERE source.name = $source AND target.name = $target "
                        "MERGE (source)-[r:`{relation}`]->(target) "
                        "SET r.value = $relation"
                    )
                    await tx.run(query, {
                        "source": relationship["source"],
                        "target": relationship["target"],
                        "relation": relationship["relation"],
                        "user_id": user_id
                    })
                    logger.debug(f"Transaction: Created relationship {relationship['source']} -> {relationship['target']}")
                
                # Step 3: Add embeddings (using the official Neo4j procedure)
                for emb_data in embeddings_data:
                    query = """
                    MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                    CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                    RETURN n.name
                    """
                    result = await tx.run(query, {
                        "node_name": emb_data["node_name"],
                        "embedding": emb_data["embedding"],
                        "user_id": user_id
                    })
                    # Consume all results to avoid warnings (may be multiple nodes with same name)
                    records = await result.data()
                    logger.debug(f"Transaction: Added embedding for {emb_data['node_name']} ({len(records)} nodes updated)")
                
                # Step 4: Update bloom levels and other properties
                # If bloom_level changed, append to bloom_history before updating
                from datetime import datetime, timezone

                for bloom_data in bloom_updates:
                    props = bloom_data.get("properties", {})
                    new_bloom_level = props.get("bloom_level", "")

                    # Initialize variables to track bloom level changes
                    check_record = None
                    current_history = []
                    bloom_level_changed = False

                    # First, check if bloom_level is changing
                    if new_bloom_level:
                        check_query = """
                        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                        RETURN n.bloom_level AS current_level, n.bloom_history AS bloom_history
                        """
                        check_result = await tx.run(check_query, {
                            "node_name": bloom_data["node_name"],
                            "user_id": user_id
                        })
                        check_record = await check_result.single()

                        if check_record:
                            current_level = check_record.get("current_level")
                            current_history_json = check_record.get("bloom_history")

                            # If bloom level is changing, append to history
                            if current_level and new_bloom_level != current_level:
                                # Parse existing bloom_history
                                current_history = []
                                if current_history_json:
                                    try:
                                        current_history = json.loads(current_history_json)
                                    except (json.JSONDecodeError, TypeError):
                                        logger.warning(f"Failed to parse existing bloom_history for {bloom_data['node_name']}")

                                # Determine source from entity IDs
                                source_info = []
                                book_ids = bloom_data.get('book_id', [])
                                highlight_ids = bloom_data.get('highlight_id', [])
                                writing_ids = bloom_data.get('writing_id', [])

                                if highlight_ids:
                                    source_info.append(f"highlight_id:{highlight_ids[0]}")
                                elif book_ids:
                                    source_info.append(f"book_id:{book_ids[0]}")
                                elif writing_ids:
                                    source_info.append(f"writing_id:{writing_ids[0]}")

                                source = ', '.join(source_info) if source_info else None

                                # Create new bloom level update entry
                                new_entry = {
                                    "level": new_bloom_level,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "source": source
                                }

                                current_history.append(new_entry)
                                bloom_level_changed = True

                                # Log the change
                                logger.info(
                                    f"Bloom level change detected for '{bloom_data['node_name']}': "
                                    f"{current_level} → {new_bloom_level}"
                                )

                    # Now update the node properties (including bloom_level and potentially bloom_history)
                    query = """
                    MATCH (n:NodeName {name: $node_name, UserId: $user_id})
                    SET n.discipline = $discipline,
                        n.bloom_level = $bloom_level,
                        n.confidence = $confidence
                    """
                    params = {
                        "node_name": bloom_data["node_name"],
                        "user_id": user_id,
                        "discipline": props.get("discipline", ""),
                        "bloom_level": new_bloom_level,
                        "confidence": props.get("confidence", 0.0)
                    }

                    # If we detected a bloom level change, update bloom_history
                    if bloom_level_changed:
                        query = query.rstrip() + ", n.bloom_history = $bloom_history\n                    "
                        params["bloom_history"] = json.dumps(current_history)

                    # Preserve chunk_ids if present
                    chunk_ids = props.get("chunk_ids")
                    if chunk_ids:
                        query = query.rstrip() + ", n.chunk_ids = $chunk_ids\n                    "
                        params["chunk_ids"] = chunk_ids

                    await tx.run(query, params)
                    logger.debug(f"Transaction: Updated properties for {bloom_data['node_name']}")
                
                # Commit all changes atomically
                await tx.commit()
                logger.info(f"Transaction committed: {len(nodes)} nodes, {len(relationships)} rels, {len(bloom_updates)} bloom updates")
                
            except Exception as e:
                await tx.rollback()
                logger.error(f"Transaction failed, rolling back all changes: {e}")
                raise
            finally:
                await tx.close()

    async def create_vector_index(self, index_name: str) -> None:
        # Check if the index already exists
        existing_indexes_query = "SHOW VECTOR INDEXES"
        async with self._ensure_driver().session() as session:
            existing_indexes = await session.run(existing_indexes_query)
            index_exists = any(index['name'] == index_name for index in await existing_indexes.data())
    
        if not index_exists:
            query = """
            CREATE VECTOR INDEX `embeddings_index`
            FOR (n:NodeName) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """
            async with self._ensure_driver().session() as session:
                await session.run(query)
            logger.info(f"Vector index '{index_name}' created.")
        else:
            logger.debug(f"Vector index '{index_name}' already exists.")

    async def add_embedding_to_vector_index(self, node_name: str, embedding: List[float], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add embedding.")
            return

        # Use the official Neo4j procedure for setting vector properties
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        RETURN n.name
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, node_name=node_name, embedding=embedding, user_id=user_id)
            # Consume all results to handle multiple nodes with same name
            records = await result.data()
            if records:
                logger.debug(f"Added embedding to {len(records)} node(s) named '{node_name}'")

    async def index_exists(self, index_name: str) -> bool:
        async with self._ensure_driver().session() as session:
            existing_indexes = await session.run("SHOW VECTOR INDEXES")
            index_exists = any(index['name'] == index_name for index in await existing_indexes.data())
            return index_exists

    async def ensure_vector_index(self) -> None:
        async with self._ensure_driver().session() as session:
            # Check if the index exists
            result = await session.run("SHOW VECTOR INDEXES")
            indexes = await result.data()
            index_exists = any(index['name'] == 'embeddings_index' for index in indexes)

            if not index_exists:
                # Create the index if it doesn't exist
                query = """
                CREATE VECTOR INDEX embeddings_index
                FOR (n:NodeName)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
                """
                try:
                    await session.run(query)
                    logger.info("Vector index 'embeddings_index' created.")
                except Exception as e:
                    if "EquivalentSchemaRuleAlreadyExists" in str(e):
                        logger.debug("Vector index 'embeddings_index' already exists (caught exception).")
                    else:
                        raise e
            else:
                logger.debug("Vector index 'embeddings_index' already exists.")

    async def query_text_similarity(
        self,
        keyword_embedding: List[float],
        user_id: str,
        limit: int = 5,
        index_name: str = "embeddings_index",
        book_id: Optional[int] = None,
        highlight_id: Optional[int] = None,
        writing_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query nodes by cosine similarity with optional entity ID pre-filtering.
        When entity IDs are provided, pre-filters nodes before computing similarity.

        Args:
        - keyword_embedding (List[float]): The embedding of the text keyword as a list of floats.
        - user_id (str): The user ID to filter the nodes by.
        - limit (int): Maximum number of results to return (default: 5).
        - index_name (str): The name of the vector index used for querying.
        - book_id (Optional[int]): Optional book ID to filter results.
        - highlight_id (Optional[int]): Optional highlight ID to filter results.
        - writing_id (Optional[int]): Optional writing ID to filter results.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries containing the node ID, node name, and their similarity scores.
        """
        # If entity IDs are provided, use direct query with pre-filtering
        if book_id is not None or highlight_id is not None or writing_id is not None:
            return await self._query_similarity_with_entity_filter(
                keyword_embedding, user_id, limit, book_id, highlight_id, writing_id
            )

        # Otherwise use the vector index (faster for unfiltered queries)
        query = """
        CALL db.index.vector.queryNodes($indexName, $limit, $embedding)
        YIELD node, score
        WHERE node.UserId = $user_id
        RETURN elementId(node) AS nodeId, node.name AS nodeName, score
        ORDER BY score DESC
        """

        results = []
        async with self._ensure_driver().session() as session:
            tx = await session.begin_transaction()
            try:
                result = await tx.run(query, indexName=index_name, embedding=keyword_embedding, user_id=user_id, limit=limit)
                async for record in result:
                    results.append({
                        'nodeId': record['nodeId'],
                        'nodeName': record['nodeName'],
                        'score': record['score']
                    })
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                raise
            finally:
                await tx.close()
        return results

    async def _query_similarity_with_entity_filter(
        self,
        keyword_embedding: List[float],
        user_id: str,
        limit: int,
        book_id: Optional[int] = None,
        highlight_id: Optional[int] = None,
        writing_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query nodes by computing cosine similarity directly, with entity ID pre-filtering.
        This avoids the vector index and allows proper filtering before similarity computation.
        """
        # Build WHERE conditions for entity ID filtering
        where_conditions = ["n.UserId = $user_id", "n.embedding IS NOT NULL"]
        params = {
            "user_id": user_id,
            "embedding": keyword_embedding,
            "limit": limit
        }

        if book_id is not None:
            where_conditions.append("$book_id IN n.book_id")
            params["book_id"] = book_id

        if highlight_id is not None:
            where_conditions.append("$highlight_id IN n.highlight_id")
            params["highlight_id"] = highlight_id

        if writing_id is not None:
            where_conditions.append("$writing_id IN n.writing_id")
            params["writing_id"] = writing_id

        where_clause = " AND ".join(where_conditions)

        # Direct cosine similarity calculation on filtered nodes
        query = f"""
        MATCH (n:NodeName)
        WHERE {where_clause}
        WITH n,
             reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
                dot + n.embedding[i] * $embedding[i]
             ) AS dotProduct,
             sqrt(reduce(s = 0.0, x IN n.embedding | s + x * x)) AS normA,
             sqrt(reduce(s = 0.0, x IN $embedding | s + x * x)) AS normB
        WITH n, dotProduct / (normA * normB) AS score
        WHERE score > 0
        RETURN elementId(n) AS nodeId, n.name AS nodeName, score
        ORDER BY score DESC
        LIMIT $limit
        """

        logger.info(f"Similarity search with entity pre-filter: book_id={book_id}, highlight_id={highlight_id}, writing_id={writing_id}")

        results = []
        async with self._ensure_driver().session() as session:
            tx = await session.begin_transaction()
            try:
                result = await tx.run(query, **params)
                async for record in result:
                    results.append({
                        'nodeId': record['nodeId'],
                        'nodeName': record['nodeName'],
                        'score': record['score']
                    })
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                raise
            finally:
                await tx.close()

        logger.info(f"Entity-filtered similarity search returned {len(results)} results")
        return results


    async def update_node_embeddings(self, node_name: str, embedding: List[float], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update embeddings.")
            return
        if not self._validate_embedding(embedding):
            logger.error(f"Invalid embedding format for node {node_name}. Embedding must be a list of floats.")
            return

        async with self._ensure_driver().session() as session:
            success_flag = await self._set_node_embedding(session=session, embedding=embedding, node_name=node_name, user_id=user_id)

    async def _set_node_embedding(self, session, embedding: List[float], node_name: str, user_id: str) -> bool:
        query = """
        MATCH (n {name: $node_name, UserId: $user_id})
        SET n.embedding = $embedding
        RETURN n.embedding IS NOT NULL AS successFlag
        """
        logger.debug(f"Setting embedding for node {node_name} with embedding size: {len(embedding)}")
        result = await session.run(query, embedding=embedding, node_name=node_name, user_id=user_id)
        result_data = await result.single()
        return result_data['successFlag'] if result_data else False

    @staticmethod
    def _validate_embedding(embedding: List[float]) -> bool:
        return isinstance(embedding, list) and all(isinstance(item, float) for item in embedding)

    async def append_chunk_ids_to_node(
        self,
        node_name: str,
        chunk_ids: List[str],
        user_id: str
    ) -> None:
        """
        Append chunk_ids to an existing node's chunk_ids array.
        This is used when merging similar nodes - we want to preserve all source chunk references.

        Args:
            node_name: Name of the node to update
            chunk_ids: List of chunk_ids to append
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot append chunk_ids.")
            return

        if not chunk_ids:
            return

        async with self._ensure_driver().session() as session:
            # Use Cypher to append to array, removing duplicates using apoc.coll.toSet or manual deduplication
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            WITH n, COALESCE(n.chunk_ids, []) as existing_ids
            WITH n, existing_ids + [id IN $new_chunk_ids WHERE NOT id IN existing_ids] as updated_ids
            SET n.chunk_ids = updated_ids
            RETURN size(n.chunk_ids) as total_chunks
            """

            result = await session.run(
                query,
                node_name=node_name,
                user_id=user_id,
                new_chunk_ids=chunk_ids
            )

            data = await result.single()
            if data:
                logger.debug(
                    f"Node '{node_name}' now has {data['total_chunks']} chunk_ids after appending {len(chunk_ids)}"
                )

    async def append_entity_ids_to_node(
        self,
        node_name: str,
        book_ids: List[int],
        highlight_ids: List[int],
        writing_ids: List[int],
        user_id: str
    ) -> None:
        """
        Append entity IDs (book_id, highlight_id, writing_id) to an existing node's arrays.
        This is used when merging similar nodes - we want to preserve all source entity references.

        Args:
            node_name: Name of the node to update
            book_ids: List of book IDs to append
            highlight_ids: List of highlight IDs to append
            writing_ids: List of writing IDs to append
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot append entity IDs.")
            return

        if not book_ids and not highlight_ids and not writing_ids:
            return

        async with self._ensure_driver().session() as session:
            # Use Cypher to append to arrays, removing duplicates
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            WITH n,
                 COALESCE(n.book_id, []) as existing_book_ids,
                 COALESCE(n.highlight_id, []) as existing_highlight_ids,
                 COALESCE(n.writing_id, []) as existing_writing_ids
            WITH n,
                 existing_book_ids + [id IN $new_book_ids WHERE NOT id IN existing_book_ids] as updated_book_ids,
                 existing_highlight_ids + [id IN $new_highlight_ids WHERE NOT id IN existing_highlight_ids] as updated_highlight_ids,
                 existing_writing_ids + [id IN $new_writing_ids WHERE NOT id IN existing_writing_ids] as updated_writing_ids
            SET n.book_id = updated_book_ids,
                n.highlight_id = updated_highlight_ids,
                n.writing_id = updated_writing_ids
            RETURN size(n.book_id) as total_book_ids,
                   size(n.highlight_id) as total_highlight_ids,
                   size(n.writing_id) as total_writing_ids
            """

            result = await session.run(
                query,
                node_name=node_name,
                user_id=user_id,
                new_book_ids=book_ids,
                new_highlight_ids=highlight_ids,
                new_writing_ids=writing_ids
            )

            data = await result.single()
            if data:
                logger.debug(
                    f"Appended entity IDs to node '{node_name}': "
                    f"{data['total_book_ids']} book_ids, "
                    f"{data['total_highlight_ids']} highlight_ids, "
                    f"{data['total_writing_ids']} writing_ids"
                )

    async def append_bloom_level_update(
        self,
        node_name: str,
        bloom_update: Dict[str, Any],
        user_id: str
    ) -> None:
        """
        Append a new Bloom level update to the node's bloom_history.
        This is used when detecting Bloom level progressions.

        Args:
            node_name: Name of the node to update
            bloom_update: Dict with 'level', 'timestamp', 'source' keys
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot append bloom level update.")
            return

        async with self._ensure_driver().session() as session:
            # Append to bloom_history array
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            WITH n, COALESCE(n.bloom_history, []) as existing_history
            SET n.bloom_history = existing_history + [$bloom_update]
            RETURN size(n.bloom_history) as total_updates
            """

            result = await session.run(
                query,
                node_name=node_name,
                user_id=user_id,
                bloom_update=bloom_update
            )

            data = await result.single()
            if data:
                logger.debug(
                    f"Appended Bloom level update to node '{node_name}': "
                    f"total history entries: {data['total_updates']}"
                )

    async def preserve_earliest_created_at(
        self,
        node_name: str,
        new_created_at: str,
        user_id: str
    ) -> None:
        """
        Ensure the node keeps the earliest created_at timestamp.
        This is used when merging nodes - we want to preserve the original learning date.

        Args:
            node_name: Name of the node to update
            new_created_at: ISO 8601 timestamp to compare
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update created_at.")
            return

        if not new_created_at:
            return

        async with self._ensure_driver().session() as session:
            # Only update if new timestamp is earlier or if created_at doesn't exist
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            WITH n,
                 CASE
                     WHEN n.created_at IS NULL THEN $new_created_at
                     WHEN $new_created_at < n.created_at THEN $new_created_at
                     ELSE n.created_at
                 END as earliest_timestamp
            SET n.created_at = earliest_timestamp
            RETURN n.created_at as final_created_at
            """

            result = await session.run(
                query,
                node_name=node_name,
                user_id=user_id,
                new_created_at=new_created_at
            )

            data = await result.single()
            if data:
                logger.debug(
                    f"Updated created_at for node '{node_name}': {data['final_created_at']}"
                )

    async def get_node_data(self, node_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN n.name AS name, n.type AS type,
               n.discipline AS discipline,
               n.bloom_level AS bloom_level,
               n.confidence AS confidence,
               n.created_at AS created_at,
               n.bloom_history AS bloom_history
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            record = await result.single()
            if record:
                # Parse bloom_history from JSON string to list of dicts
                bloom_history_str = record.get("bloom_history")
                bloom_history = []
                if bloom_history_str:
                    try:
                        bloom_history = json.loads(bloom_history_str)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse bloom_history for node {node_name}")

                return {
                    "name": record["name"],
                    "type": record["type"],
                    "properties": {
                        "discipline": record.get("discipline", ""),
                        "bloom_level": record.get("bloom_level", ""),
                        "confidence": record.get("confidence", 0.0)
                    },
                    "created_at": record.get("created_at"),
                    "bloom_history": bloom_history
                }
            return None

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})-[r]-(m:NodeName)
        RETURN type(r) AS relation, m.name AS related_node, r.value AS value,
               CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            return [
                {
                    "source": node_name if record["direction"] == "outgoing" else record["related_node"],
                    "target": record["related_node"] if record["direction"] == "outgoing" else node_name,
                    "relation": record["relation"],
                    "value": record["value"]
                } 
                for record in await result.data()
            ]

    async def get_all_nodes(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {UserId: $user_id})
        RETURN n.name AS name, n.type AS type,
               n.discipline AS discipline,
               n.bloom_level AS bloom_level,
               n.confidence AS confidence,
               n.created_at AS created_at,
               n.bloom_history AS bloom_history
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, user_id=user_id)
            data = await result.data()
            # Build properties dict from individual fields
            for record in data:
                record['properties'] = {
                    "discipline": record.get('discipline', ""),
                    "bloom_level": record.get('bloom_level', ""),
                    "confidence": record.get('confidence', 0.0)
                }
                # Keep temporal fields at top level
                record['created_at'] = record.get('created_at')

                # Parse bloom_history from JSON string to list of dicts
                bloom_history_str = record.get('bloom_history')
                bloom_history = []
                if bloom_history_str:
                    try:
                        bloom_history = json.loads(bloom_history_str)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse bloom_history for node {record.get('name')}")
                record['bloom_history'] = bloom_history

                # Remove individual fields from top level
                record.pop('discipline', None)
                record.pop('bloom_level', None)
                record.pop('confidence', None)
            return data

    async def get_all_relationships(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (source:NodeName {UserId: $user_id})-[r]->(target:NodeName {UserId: $user_id})
        RETURN source.name AS source, type(r) AS relation, target.name AS target
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, user_id=user_id)
            return await result.data()

    async def create_user(self, user_id: str) -> None:
        query = """
        MERGE (u:User {id: $user_id})
        """
        logger.debug(f"Creating user {user_id} with URI: {self.uri}")
        async with self._ensure_driver().session() as session:
            await session.run(query, user_id=user_id)
        logger.info(f"User {user_id} created successfully.")

    async def user_exists(self, user_id: str) -> bool:
        query = """
        MATCH (u:User {id: $user_id})
        RETURN COUNT(u) > 0 AS exists
        """
        async with self._ensure_driver().session() as session:
            result = await session.run(query, user_id=user_id)
            record = await result.single()
            return bool(record['exists']) if record else False

    async def delete_user(self, user_id: str) -> None:
        # First delete all nodes associated with the user
        query1 = """
        MATCH (n {UserId: $user_id})
        DETACH DELETE n
        """
        # Then delete the user node itself
        query2 = """
        MATCH (u:User {id: $user_id})
        DELETE u
        """
        async with self._ensure_driver().session() as session:
            await session.run(query1, user_id=user_id)
            await session.run(query2, user_id=user_id)
        logger.info(f"User {user_id} and all associated nodes deleted successfully.")


