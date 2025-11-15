from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.llm.llm_graph import get_nodes, get_relationships, assess_cognitive_level, Node as LLMNode
from persona.llm.embeddings import generate_embeddings
from persona.models.schema import (
    NodeModel, RelationshipModel, GraphUpdateModel,
    UnstructuredData, NodesAndRelationshipsResponse, Node, Relationship
)
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from server.logging_config import get_logger

logger = get_logger(__name__)

class GraphConstructor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None
        self.graph_context_retriever = None

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.graph_ops is not None:
            await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)

    async def clean_graph(self):
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        await self.graph_ops.clean_graph()

    async def _generate_all_relationships(
        self,
        nodes: List[Node],
        text: str,
        book_ids: set[int]
    ) -> List[Relationship]:
        """
        Generate all types of relationships for the given nodes.

        Args:
            nodes: List of nodes to generate relationships for
            text: Original text for context retrieval
            book_ids: Set of book IDs for book-scoped relationships

        Returns:
            List of all relationships
        """
        if self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        relationships = []

        # OPTIMIZATION: Filter out CognitiveLevel nodes from relationship generation
        # CognitiveLevel nodes can ONLY have HAS_UNDERSTANDING_LEVEL (created by system, not LLM)
        # This saves LLM tokens and processing time
        nodes_for_llm = [n for n in nodes if n.type != "CognitiveLevel"]

        if len(nodes_for_llm) < len(nodes):
            logger.info(
                f"Filtered out {len(nodes) - len(nodes_for_llm)} CognitiveLevel node(s) "
                f"from relationship generation (they can only have HAS_UNDERSTANDING_LEVEL)"
            )

        # Get existing graph context
        existing_context = await self.graph_context_retriever.get_rich_context(text, self.user_id)

        # Phase 1: Core relationships between new nodes
        new_node_relationships = await self.generate_relationships(nodes_for_llm)
        relationships.extend(new_node_relationships)

        # Phase 2: Connect with existing nodes
        if existing_context and len(nodes_for_llm) > 0:
            mixed_relationships = await self.generate_cross_relationships(nodes_for_llm, existing_context)
            relationships.extend(mixed_relationships)

        # Phase 3: Book-scoped relationships
        for book_id in book_ids:
            book_relationships = await self.generate_book_scoped_relationships(nodes_for_llm, book_id)
            relationships.extend(book_relationships)
            logger.info(f"Created {len(book_relationships)} book-scoped relationships for book {book_id}")

        # Phase 4: Contrast/semantic relationships (finds philosophical opposites and debates)
        logger.info(f"Phase 4 check: book_ids={book_ids}, has_context={bool(book_ids)}")
        if book_ids:  # Only run if we have book context
            logger.info(f"Phase 4: Calling generate_contrast_relationships with {len(nodes_for_llm)} nodes and book_ids={book_ids}")
            contrast_relationships = await self.generate_contrast_relationships(nodes_for_llm, book_ids)
            relationships.extend(contrast_relationships)
            logger.info(f"Created {len(contrast_relationships)} contrast/semantic relationships")
        else:
            logger.warning("Phase 4: Skipped - no book_ids provided")

        return relationships

    def _nodes_to_node_models(self, nodes: List[Node], embeddings: List[List[float]]) -> List[NodeModel]:
        """
        Convert schema Nodes to NodeModels with embeddings.

        Args:
            nodes: List of schema Node objects
            embeddings: List of embedding vectors

        Returns:
            List of NodeModel objects ready for database
        """
        from persona.models.schema import NodeModel

        node_models = []
        for node, embedding in zip(nodes, embeddings):
            properties = {}
            if node.discipline:
                properties["discipline"] = node.discipline
            if node.confidence is not None:
                properties["confidence"] = node.confidence

            # Add concept_uuid if present (for CognitiveLevel nodes and Concept nodes)
            if hasattr(node, 'concept_uuid') and node.concept_uuid:
                properties["concept_uuid"] = node.concept_uuid

            # Convert datetime to ISO string
            created_at_str = None
            if hasattr(node, 'created_at') and node.created_at:
                created_at_str = node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at)

            node_models.append(NodeModel(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids if node.chunk_ids else [],
                book_id=node.book_id if node.book_id else [],
                highlight_id=node.highlight_id if node.highlight_id else [],
                writing_id=node.writing_id if node.writing_id else [],
                properties=properties,
                embedding=embedding,
                created_at=created_at_str
            ))

        return node_models

    def _schema_nodes_to_llm_nodes(self, nodes: List[Node]) -> List[LLMNode]:
        """
        Convert schema Node objects to LLM Node objects for relationship generation.

        Args:
            nodes: List of schema Node objects

        Returns:
            List of LLMNode objects
        """
        return [
            LLMNode(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids,
                book_id=node.book_id,
                highlight_id=node.highlight_id,
                writing_id=node.writing_id,
                discipline=node.discipline,
                confidence=node.confidence,
                source_index=getattr(node, 'source_index', None)
            ) for node in nodes
        ]

    def _llm_relationships_to_schema(self, llm_relationships) -> List[Relationship]:
        """
        Convert LLM relationships to schema relationships.

        Args:
            llm_relationships: List of LLM relationship objects

        Returns:
            List of schema Relationship objects
        """
        return [
            Relationship(source=rel.source, target=rel.target, relation=rel.relation)
            for rel in llm_relationships
        ]

    async def _get_or_generate_concept_uuid(self, concept_name: str) -> Optional[str]:
        """
        Get existing UUID for a Concept node from database, or generate a new one.

        For new Concepts that haven't been saved yet, generates a UUID in memory.
        For existing Concepts, retrieves the UUID from the database.

        Args:
            concept_name: Name of the concept node

        Returns:
            UUID string for the concept
        """
        if self.graph_ops is None:
            return None

        # Try to get existing UUID from database
        existing_uuid = await self.graph_ops._get_concept_uuid_if_exists(concept_name, self.user_id)

        if existing_uuid:
            return existing_uuid

        # Generate new UUID in memory for new Concept nodes
        import uuid
        return str(uuid.uuid4())

    async def _ensure_concepts_have_cognitive_levels(self, nodes: List[Node], has_source_index: bool) -> None:
        """
        Validation #3: Ensure every Concept node has a corresponding CognitiveLevel node.
        If a Concept is missing a CognitiveLevel, create a default "Remember" node IN MEMORY.

        This MUST run BEFORE cognitive assessment so that default nodes also get assessed.

        Args:
            nodes: List of all nodes
            has_source_index: Whether this is batch mode (True) or single mode (False)
        """
        from collections import defaultdict

        if has_source_index:
            # Batch mode: Group by source_index
            source_groups = defaultdict(lambda: {'concepts': [], 'cognitive_levels': []})
            for node in nodes:
                source_idx = getattr(node, 'source_index', None)
                if source_idx is None:
                    continue
                if node.type == "Concept":
                    source_groups[source_idx]['concepts'].append(node)
                elif node.type == "CognitiveLevel":
                    source_groups[source_idx]['cognitive_levels'].append(node)

            # Check each source_index
            for source_idx, group in source_groups.items():
                if group['concepts'] and not group['cognitive_levels']:
                    # Concept exists but no CognitiveLevel - create default
                    logger.warning(
                        f"Source [{source_idx}] has Concept but no CognitiveLevel - creating default 'Remember'"
                    )
                    from persona.models.schema import Node as SchemaNode
                    from datetime import datetime, timezone
                    default_cognitive_level = SchemaNode(
                        name="Remember",
                        type="CognitiveLevel",
                        discipline="Education",
                        confidence=0.5,
                        created_at=datetime.now(timezone.utc)
                    )
                    default_cognitive_level.source_index = source_idx  # type: ignore
                    nodes.append(default_cognitive_level)
        else:
            # Single mode: Check all Concepts
            concepts = [n for n in nodes if n.type == "Concept"]
            cognitive_levels = [n for n in nodes if n.type == "CognitiveLevel"]

            # If there are Concepts but no CognitiveLevels, create defaults
            if concepts and not cognitive_levels:
                logger.warning(
                    f"Found {len(concepts)} Concept(s) but no CognitiveLevel nodes - creating defaults"
                )
                from persona.models.schema import Node as SchemaNode
                from datetime import datetime, timezone

                # Create one default "Remember" CognitiveLevel per Concept
                # (Following the same pattern as batch mode, but for single mode)
                for concept in concepts:
                    default_cognitive_level = SchemaNode(
                        name="Remember",
                        type="CognitiveLevel",
                        discipline="Education",
                        confidence=0.5,
                        created_at=datetime.now(timezone.utc)
                    )
                    nodes.append(default_cognitive_level)
                    logger.debug(f"Created default 'Remember' CognitiveLevel for Concept '{concept.name}'")

    async def _get_existing_cognitive_level_values(self, concept_uuid: str) -> set:
        """
        Query existing CognitiveLevel values connected to a Concept via shared UUID.

        Args:
            concept_uuid: UUID of the concept node

        Returns:
            Set of cognitive level values (e.g., {"Remember", "Understand"})
        """
        if self.graph_ops is None:
            return set()
        existing_levels = await self.graph_ops._get_concept_cognitive_levels_by_uuid(concept_uuid, self.user_id)

        # Extract level names directly (no parsing needed with UUID-based approach)
        return {level_data["level"] for level_data in existing_levels}

    async def _create_cognitive_level_relationships(self, nodes: List[Node]) -> List[Relationship]:
        """
        Create HAS_UNDERSTANDING_LEVEL relationships between Concepts and CognitiveLevels.

        IMPORTANT: Only Concept nodes (type="Concept") get CognitiveLevel relationships.
        Term nodes, Person nodes, and other types do NOT receive cognitive level tracking.

        For batch ingestion: Groups nodes by source_index and creates one relationship
        per source_index connecting the Concept to its CognitiveLevel.

        For single ingestion: Creates relationships between all Concepts and CognitiveLevels
        (since source_index will be None for all nodes).

        Validation Rules:
        1. Each CognitiveLevel node can have ONLY ONE outgoing edge (to its parent Concept)
        2. A Concept cannot have duplicate CognitiveLevel values (e.g., can't connect to two "Remember" nodes)
        3. Each Concept node MUST have at least one connection to a CognitiveLevel node

        Args:
            nodes: List of schema Node objects

        Returns:
            List of Relationship objects with HAS_UNDERSTANDING_LEVEL relations
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        relationships = []

        # Track which CognitiveLevel nodes are being connected (validation rule #1)
        # IMPORTANT: Track by node instance ID, not by name (multiple "Remember" nodes are allowed)
        cognitive_level_connections = {}  # id(cognitive_level) -> concept.name

        # Track which Concepts have received relationships (validation rule #3)
        concepts_with_relationships = set()

        # Check if we have source_index data (batch ingestion)
        has_source_index = any(getattr(node, 'source_index', None) is not None for node in nodes)

        # VALIDATION #3 FIRST: Ensure every Concept has a CognitiveLevel node BEFORE assessment
        # This must run BEFORE assessment so default nodes get assessed too
        await self._ensure_concepts_have_cognitive_levels(nodes, has_source_index)

        if has_source_index:
            # Batch mode: Group by source_index
            from collections import defaultdict
            source_groups = defaultdict(lambda: {'concepts': [], 'cognitive_levels': [], 'usernotes': []})

            for node in nodes:
                source_idx = getattr(node, 'source_index', None)
                if source_idx is None:
                    continue

                if node.type == "Concept":
                    source_groups[source_idx]['concepts'].append(node)
                elif node.type == "CognitiveLevel":
                    source_groups[source_idx]['cognitive_levels'].append(node)
                elif node.type == "UserNote":
                    source_groups[source_idx]['usernotes'].append(node)

            # Create one relationship per source_index
            for source_idx, group in source_groups.items():
                concepts = group['concepts']
                cognitive_levels = group['cognitive_levels']
                usernotes = group['usernotes']

                if not concepts or not cognitive_levels:
                    if concepts and not cognitive_levels:
                        logger.warning(f"Source [{source_idx}] has Concept(s) but no CognitiveLevel")
                    elif cognitive_levels and not concepts:
                        logger.warning(f"Source [{source_idx}] has CognitiveLevel(s) but no Concept")
                    continue

                # Take first Concept and first CognitiveLevel for this source
                concept = concepts[0]
                cognitive_level = cognitive_levels[0]

                # ASSESSMENT PHASE: Assess cognitive level based on UserNote
                if usernotes:
                    usernote = usernotes[0]
                    try:
                        assessed_level = await assess_cognitive_level(usernote.name)

                        if assessed_level != cognitive_level.name:
                            logger.info(
                                f"Cognitive level assessment corrected: '{cognitive_level.name}' → '{assessed_level}' "
                                f"for UserNote: '{usernote.name[:50]}...'"
                            )
                            # Update the cognitive level node's name with assessed level
                            cognitive_level.name = assessed_level
                        else:
                            logger.debug(f"Cognitive level assessment confirmed: '{assessed_level}'")
                    except Exception as e:
                        logger.error(f"Error during cognitive assessment for source [{source_idx}]: {e}")
                        # Continue with original level if assessment fails
                else:
                    logger.warning(f"Source [{source_idx}] has no UserNote for cognitive assessment")

                # Get or generate UUID for the Concept
                concept_uuid = await self._get_or_generate_concept_uuid(concept.name)
                if not concept_uuid:
                    logger.error(f"Failed to get UUID for Concept '{concept.name}'")
                    continue

                # Add UUID to Concept node for database storage
                concept.concept_uuid = concept_uuid  # type: ignore

                # Store the level value
                level_value = cognitive_level.name

                # Validation #2: Concept cannot have duplicate CognitiveLevel values
                existing_values = await self._get_existing_cognitive_level_values(concept_uuid)
                if level_value in existing_values:
                    logger.error(
                        f"Validation failed: Concept '{concept.name}' already connected to "
                        f"CognitiveLevel value '{level_value}'"
                    )
                    continue

                # Validation #1: CognitiveLevel can only connect to ONE Concept
                # Track using node instance ID (multiple "Remember" nodes are allowed, but each instance connects once)
                cognitive_level_id = id(cognitive_level)
                if cognitive_level_id in cognitive_level_connections:
                    logger.error(
                        f"Validation failed: CognitiveLevel '{cognitive_level.name}' (instance {cognitive_level_id}) already connected to "
                        f"'{cognitive_level_connections[cognitive_level_id]}', cannot connect to '{concept.name}'"
                    )
                    continue

                # Add shared UUID to CognitiveLevel node
                cognitive_level.concept_uuid = concept_uuid  # type: ignore

                # Track this connection by node instance
                cognitive_level_connections[cognitive_level_id] = concept.name
                concepts_with_relationships.add(concept.name)

                relationships.append(Relationship(
                    source=concept.name,
                    target=cognitive_level.name,
                    relation="HAS_UNDERSTANDING_LEVEL"
                ))

                if len(concepts) > 1:
                    logger.warning(f"Source [{source_idx}] has {len(concepts)} Concepts, using first: {concept.name}")
                if len(cognitive_levels) > 1:
                    logger.warning(f"Source [{source_idx}] has {len(cognitive_levels)} CognitiveLevels, using first: {cognitive_level.name}")

        else:
            # Single insert mode: Connect all Concepts to all CognitiveLevels
            concepts = [n for n in nodes if n.type == "Concept"]
            cognitive_levels = [n for n in nodes if n.type == "CognitiveLevel"]
            usernotes = [n for n in nodes if n.type == "UserNote"]

            # ASSESSMENT PHASE: Assess cognitive levels based on UserNotes (single mode)
            if usernotes and cognitive_levels:
                # Typically 1 UserNote and 1 CognitiveLevel in single insert mode
                for i, cognitive_level in enumerate(cognitive_levels):
                    if i < len(usernotes):  # Match UserNote to CognitiveLevel by index
                        usernote = usernotes[i]
                        try:
                            assessed_level = await assess_cognitive_level(usernote.name)

                            if assessed_level != cognitive_level.name:
                                logger.info(
                                    f"Cognitive level assessment corrected: '{cognitive_level.name}' → '{assessed_level}' "
                                    f"for UserNote: '{usernote.name[:50]}...'"
                                )
                                # Update the cognitive level node's name with assessed level
                                cognitive_level.name = assessed_level
                            else:
                                logger.debug(f"Cognitive level assessment confirmed: '{assessed_level}'")
                        except Exception as e:
                            logger.error(f"Error during cognitive assessment in single mode: {e}")
                            # Continue with original level if assessment fails

            if concepts and cognitive_levels:
                # For single insert, typically expect 1 Concept and 1 CognitiveLevel
                for concept in concepts:
                    # Get or generate UUID for the Concept
                    concept_uuid = await self._get_or_generate_concept_uuid(concept.name)
                    if not concept_uuid:
                        logger.error(f"Failed to get UUID for Concept '{concept.name}'")
                        continue

                    # Add UUID to Concept node for database storage
                    concept.concept_uuid = concept_uuid  # type: ignore

                    for cognitive_level in cognitive_levels:
                        # Store the level value
                        level_value = cognitive_level.name

                        # Validation #2: Concept cannot have duplicate CognitiveLevel values
                        existing_values = await self._get_existing_cognitive_level_values(concept_uuid)
                        if level_value in existing_values:
                            logger.error(
                                f"Validation failed: Concept '{concept.name}' already connected to "
                                f"CognitiveLevel value '{level_value}'"
                            )
                            continue

                        # Validation #1: CognitiveLevel can only connect to ONE Concept
                        # Track using node instance ID (multiple "Remember" nodes are allowed, but each instance connects once)
                        cognitive_level_id = id(cognitive_level)
                        if cognitive_level_id in cognitive_level_connections:
                            logger.error(
                                f"Validation failed: CognitiveLevel '{cognitive_level.name}' (instance {cognitive_level_id}) already connected to "
                                f"'{cognitive_level_connections[cognitive_level_id]}', cannot connect to '{concept.name}'"
                            )
                            continue

                        # Add shared UUID to CognitiveLevel node
                        cognitive_level.concept_uuid = concept_uuid  # type: ignore

                        # Track this connection by node instance
                        cognitive_level_connections[cognitive_level_id] = concept.name
                        concepts_with_relationships.add(concept.name)

                        relationships.append(Relationship(
                            source=concept.name,
                            target=cognitive_level.name,
                            relation="HAS_UNDERSTANDING_LEVEL"
                        ))

        # Validation #3: Each Concept MUST have at least one CognitiveLevel relationship
        # If a Concept has no CognitiveLevel, create a default "Remember" level
        all_concepts = [n for n in nodes if n.type == "Concept"]
        for concept in all_concepts:
            # Check if concept received a relationship in this batch
            if concept.name not in concepts_with_relationships:
                # Get or generate UUID for the concept
                concept_uuid = await self._get_or_generate_concept_uuid(concept.name)
                if not concept_uuid:
                    logger.error(f"Failed to get UUID for Concept '{concept.name}' during validation")
                    continue

                # Add UUID to Concept node for database storage
                concept.concept_uuid = concept_uuid  # type: ignore

                # Check if concept has any existing relationships in the database
                existing_values = await self._get_existing_cognitive_level_values(concept_uuid)
                if not existing_values:
                    logger.warning(
                        f"Concept '{concept.name}' has no CognitiveLevel relationship. "
                        f"Creating default 'Remember' level."
                    )

                    # Create a default CognitiveLevel node and relationship
                    from datetime import datetime, timezone
                    default_level_name = "Remember"

                    # Create the default CognitiveLevel node in the database
                    await self.graph_ops._create_cognitive_level_node_for_concept(
                        concept_name=concept.name,
                        cognitive_level=default_level_name,
                        user_id=self.user_id,
                        created_at=datetime.now(timezone.utc).isoformat()
                    )

                    logger.info(f"Created default CognitiveLevel '{default_level_name}' for Concept '{concept.name}'")

        if relationships:
            logger.info(f"Created {len(relationships)} HAS_UNDERSTANDING_LEVEL relationship(s)")

        return relationships

    async def _create_highlight_usernote_relationships(self, nodes: List[Node]) -> List[Relationship]:
        """
        Create ANNOTATED_WITH relationships between Highlight and UserNote nodes.

        Rule: Every UserNote MUST have a corresponding Highlight node, and they must be connected.
        Note: Not every Highlight has a UserNote (user can highlight without adding a note).

        This method ensures that when both Highlight and UserNote nodes are present in the same
        ingestion batch (identified by matching highlight_id), they are automatically connected
        with an ANNOTATED_WITH relationship.

        Args:
            nodes: List of schema Node objects

        Returns:
            List of Relationship objects with ANNOTATED_WITH relations
        """
        relationships = []

        # Find all Highlight and UserNote nodes
        highlights = [n for n in nodes if n.type == "Highlight"]
        usernotes = [n for n in nodes if n.type == "UserNote"]

        # If no UserNotes, nothing to do
        if not usernotes:
            return relationships

        # Create a mapping of highlight_id -> Highlight nodes
        # highlight_id is stored as a list, so we need to check the first element
        highlight_map = {}
        for highlight in highlights:
            if highlight.highlight_id and len(highlight.highlight_id) > 0:
                highlight_id = highlight.highlight_id[0]
                highlight_map[highlight_id] = highlight

        # For each UserNote, find its matching Highlight and create relationship
        for usernote in usernotes:
            if not usernote.highlight_id or len(usernote.highlight_id) == 0:
                logger.error(f"UserNote '{usernote.name}' has no highlight_id - this violates the constraint that every UserNote must have a Highlight")
                continue

            highlight_id = usernote.highlight_id[0]

            # Find the matching Highlight node
            if highlight_id not in highlight_map:
                logger.error(
                    f"UserNote '{usernote.name}' references highlight_id={highlight_id}, "
                    f"but no matching Highlight node found in this batch. This violates the constraint "
                    f"that every UserNote must have a corresponding Highlight."
                )
                continue

            highlight = highlight_map[highlight_id]

            # Create the ANNOTATED_WITH relationship: Highlight -> UserNote
            relationships.append(Relationship(
                source=highlight.name,
                target=usernote.name,
                relation="ANNOTATED_WITH"
            ))

            logger.debug(f"Created ANNOTATED_WITH: '{highlight.name}' -> '{usernote.name}'")

        if relationships:
            logger.info(f"Created {len(relationships)} ANNOTATED_WITH relationship(s)")

        return relationships

    async def _recalculate_bloom_levels(self, concept_names: List[str]) -> None:
        """
        Recalculate bloom levels for given concepts after ingestion.

        This checks if any concepts have progressed to higher cognitive levels
        and creates new CognitiveLevel nodes if needed.

        Args:
            concept_names: List of concept node names to recalculate
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        try:
            stats = await self.graph_ops.recalculate_bloom_levels_for_concepts(
                concept_names=concept_names,
                user_id=self.user_id
            )

            if stats["progressions"]:
                logger.info(
                    f"Bloom level progression detected: {len(stats['progressions'])} concept(s) advanced"
                )
                for progression in stats["progressions"]:
                    logger.debug(
                        f"  - {progression['concept']}: {progression['old_level']} → {progression['new_level']}"
                    )
        except Exception as e:
            # Don't fail the entire ingestion if bloom recalculation has issues
            logger.warning(f"Bloom level recalculation failed (non-fatal): {e}")

    async def _save_graph_update(self, nodes: List[Node], relationships: List[Relationship]):
        """
        Generate embeddings and save nodes and relationships to graph.

        Args:
            nodes: List of schema Node objects
            relationships: List of Relationship objects
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        from persona.models.schema import RelationshipModel, NodesAndRelationshipsResponse

        # IMPORTANT: Create system-managed relationships BEFORE NodeModel conversion
        # This allows cognitive level assessment to update node names before they're frozen in NodeModels

        # Get all CognitiveLevel node names to filter relationships
        cognitive_level_names = {node.name for node in nodes if node.type == "CognitiveLevel"}

        # Filter out any system-managed relationships from LLM (these should only be created by the system)
        # - HAS_UNDERSTANDING_LEVEL: System creates these for Concept-CognitiveLevel pairs
        # - ANNOTATED_WITH: System creates these for Highlight-UserNote pairs
        # CRITICAL: Also filter out ANY relationship involving CognitiveLevel nodes
        # CognitiveLevel nodes can ONLY have ONE edge: HAS_UNDERSTANDING_LEVEL from their parent Concept
        llm_relationships = [
            rel for rel in relationships
            if rel.relation not in ["HAS_UNDERSTANDING_LEVEL", "ANNOTATED_WITH"]
            and rel.source not in cognitive_level_names  # CognitiveLevel cannot be source
            and rel.target not in cognitive_level_names  # CognitiveLevel cannot be target
        ]

        filtered_count = len(relationships) - len(llm_relationships)
        if filtered_count > 0:
            # Log specific reasons for filtering
            for rel in relationships:
                if rel not in llm_relationships:
                    if rel.relation in ["HAS_UNDERSTANDING_LEVEL", "ANNOTATED_WITH"]:
                        logger.debug(f"Filtered system-managed relationship: {rel.source} -{rel.relation}-> {rel.target}")
                    elif rel.source in cognitive_level_names or rel.target in cognitive_level_names:
                        logger.warning(
                            f"Filtered invalid relationship involving CognitiveLevel: "
                            f"{rel.source} -{rel.relation}-> {rel.target} "
                            f"(CognitiveLevel nodes can only have HAS_UNDERSTANDING_LEVEL from parent Concept)"
                        )
            logger.info(f"Filtered out {filtered_count} invalid/system-managed relationship(s)")

        # Create system-managed relationships (this includes cognitive level assessment)
        cognitive_relationships = await self._create_cognitive_level_relationships(nodes)
        highlight_usernote_relationships = await self._create_highlight_usernote_relationships(nodes)

        # NOW generate embeddings and convert to NodeModels (after cognitive assessment updated node names)
        node_texts = [node.name for node in nodes]
        embeddings = generate_embeddings(node_texts)

        # Convert to NodeModels
        node_models = self._nodes_to_node_models(nodes, embeddings)

        # Convert relationships (include both LLM-generated and system-managed relationships)
        all_relationships = llm_relationships + cognitive_relationships + highlight_usernote_relationships

        # DEDUPLICATION: Remove generic RELATED_TO edges when multiple edges exist between same nodes
        # Build a map of (source, target) pairs to all their relationships
        relationship_map = {}
        for rel in all_relationships:
            key = (rel.source, rel.target)
            if key not in relationship_map:
                relationship_map[key] = []
            relationship_map[key].append(rel)

        # Enhanced deduplication: Remove duplicates AND remove RELATED_TO when more specific relationships exist
        deduplicated_relationships = []
        seen_relationships = set()  # Track (source, target, relation) to prevent duplicates
        removed_duplicates = 0
        removed_related_to = 0
        removed_bidirectional = 0

        for rel in all_relationships:
            # Check for exact duplicates (same source, target, and relation type)
            rel_key = (rel.source, rel.target, rel.relation)
            if rel_key in seen_relationships:
                logger.info(
                    f"Removing duplicate relationship: '{rel.source}' -{rel.relation}-> '{rel.target}'"
                )
                removed_duplicates += 1
                continue

            seen_relationships.add(rel_key)

            # Also check if this is RELATED_TO and there are more specific relationships
            if rel.relation == 'RELATED_TO':
                key = (rel.source, rel.target)
                relations_for_pair = relationship_map[key]
                other_relations = {r.relation for r in relations_for_pair if r.relation != 'RELATED_TO'}

                if other_relations:
                    logger.info(
                        f"Removing redundant RELATED_TO: '{rel.source}' -> '{rel.target}' "
                        f"(has more meaningful: {other_relations})"
                    )
                    removed_related_to += 1
                    continue

                # Check for bidirectional RELATED_TO (A->B and B->A both exist)
                # Keep only one direction (alphabetically first) to avoid redundancy
                reverse_key = (rel.target, rel.source, 'RELATED_TO')
                if reverse_key in seen_relationships:
                    # Both directions exist, keep only alphabetically first
                    if rel.source > rel.target:
                        logger.info(
                            f"Removing bidirectional RELATED_TO: '{rel.source}' <-> '{rel.target}' "
                            f"(keeping opposite direction)"
                        )
                        removed_bidirectional += 1
                        continue

            deduplicated_relationships.append(rel)

        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate relationship(s)")
        if removed_related_to > 0:
            logger.info(f"Removed {removed_related_to} redundant RELATED_TO relationship(s)")
        if removed_bidirectional > 0:
            logger.info(f"Removed {removed_bidirectional} bidirectional RELATED_TO relationship(s)")

        # Filter out self-referencing relationships (where source == target)
        valid_relationships = []
        self_ref_count = 0
        for rel in deduplicated_relationships:
            if rel.source == rel.target:
                logger.warning(f"Filtering out self-referencing relationship: '{rel.source}' {rel.relation} '{rel.target}'")
                self_ref_count += 1
            else:
                valid_relationships.append(rel)

        if self_ref_count > 0:
            logger.warning(f"Filtered out {self_ref_count} self-referencing relationship(s)")

        relationship_models = [RelationshipModel(
            source=rel.source,
            target=rel.target,
            relation=rel.relation
        ) for rel in valid_relationships]

        graph_update = NodesAndRelationshipsResponse(
            nodes=node_models,
            relationships=relationship_models
        )

        # Save to database
        try:
            await self.graph_ops.update_graph_transactional(graph_update, self.user_id)
            logger.info(f"Successfully saved {len(node_models)} nodes and {len(relationship_models)} relationships")

            # After successful save and merging, cleanup self-referencing relationships
            deleted_count = await self.graph_ops.neo4j_manager.delete_self_referencing_relationships(self.user_id)
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} self-referencing relationship(s)")

            # Cleanup redundant RELATED_TO relationships when more specific edges exist
            related_to_deleted = await self.graph_ops.neo4j_manager.delete_redundant_related_to_relationships(self.user_id)
            if related_to_deleted > 0:
                logger.info(f"Cleaned up {related_to_deleted} redundant RELATED_TO relationship(s)")

            # Cleanup empty/placeholder UserNote nodes
            usernote_deleted = await self.graph_ops.neo4j_manager.delete_empty_usernote_nodes(self.user_id)
            if usernote_deleted > 0:
                logger.info(f"Cleaned up {usernote_deleted} empty/placeholder UserNote node(s)")

            # After successful save, recalculate bloom levels for all Concept nodes
            concept_names = [node.name for node in nodes if node.type == "Concept"]
            if concept_names:
                logger.debug(f"Recalculating bloom levels for {len(concept_names)} concept(s)")
                await self._recalculate_bloom_levels(concept_names)

        except Exception as e:
            logger.error(f"Failed to save graph update (transaction rolled back): {e}")
            raise

    def _llm_node_to_schema_node(self, llm_node, metadata: Dict[str, Any]) -> Node:
        """
        Convert LLM node to schema Node with metadata applied.

        Args:
            llm_node: Node from LLM response
            metadata: Metadata dict to apply to node

        Returns:
            Schema Node object
        """
        from datetime import datetime
        from persona.models.schema import Node

        # Extract entity IDs from metadata
        book_ids = [int(metadata['book_id'])] if 'book_id' in metadata else []
        highlight_ids = [int(metadata['highlight_id'])] if 'highlight_id' in metadata else []
        writing_ids = [int(metadata['writing_id'])] if 'writing_id' in metadata else []

        # Extract date from metadata
        annotation_date = None
        if 'date' in metadata:
            try:
                date_value = metadata['date']
                if isinstance(date_value, str):
                    annotation_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                elif isinstance(date_value, datetime):
                    annotation_date = date_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date from metadata: {e}")

        current_time = datetime.utcnow()

        node = Node(
            name=llm_node.name,
            type=llm_node.type,
            chunk_ids=getattr(llm_node, 'chunk_ids', []),
            book_id=book_ids,
            highlight_id=highlight_ids,
            writing_id=writing_ids,
            discipline=getattr(llm_node, 'discipline', ''),
            confidence=getattr(llm_node, 'confidence', 0.0),
            created_at=annotation_date or current_time
        )

        # Preserve source_index from LLM node (used for batch processing)
        if hasattr(llm_node, 'source_index') and llm_node.source_index is not None:
            node.source_index = llm_node.source_index  # type: ignore

        return node

    async def ingest_batch_unstructured_data_to_graph(self, data_items: List[UnstructuredData]):
        """
        Ingest batch of unstructured data using source indexing for accurate metadata mapping.

        Args:
            data_items: List of UnstructuredData items to ingest
        """
        if self.graph_ops is None or self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        if not data_items:
            logger.info("No data items provided for batch ingestion")
            return

        logger.info(f"Starting batch ingestion of {len(data_items)} items")

        # Format batch with source indices
        formatted_text, metadata_mapping = self.preprocess_batch_data(data_items)

        # Extract nodes (single LLM call)
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=[])
        llm_nodes = await get_nodes(formatted_text, graph_context)

        if not llm_nodes:
            logger.info("No nodes extracted from batch data")
            return

        # Theme consolidation handled by two-tier system:
        # - Tier 1: Similarity merging (automatic during graph save)
        # - Tier 2: Frequency-based pruning (after ingestion completes)

        # Map source_index to metadata and convert to schema Nodes
        schema_nodes = []
        for llm_node in llm_nodes:
            source_idx = getattr(llm_node, 'source_index', None)

            # Get metadata for this source
            # Handle single int, list of ints, or missing source_index
            if source_idx is not None and isinstance(source_idx, int) and source_idx in metadata_mapping:
                # Single source
                metadata = metadata_mapping[source_idx]
            elif source_idx is not None and isinstance(source_idx, list) and len(source_idx) > 0:
                # Cross-source node (e.g., Theme appearing in multiple chunks)
                # Use metadata from first source, but accumulate chunk_ids from all sources
                first_idx = source_idx[0]
                if first_idx in metadata_mapping:
                    metadata = metadata_mapping[first_idx].copy()
                    # Accumulate chunk_ids from all sources
                    all_chunk_ids = []
                    for idx in source_idx:
                        if idx in metadata_mapping and 'chunk_id' in metadata_mapping[idx]:
                            all_chunk_ids.append(metadata_mapping[idx]['chunk_id'])
                    if all_chunk_ids:
                        metadata['chunk_id'] = all_chunk_ids  # Will be converted to chunk_ids array
                else:
                    logger.error(f"Node '{llm_node.name}' has source_index {source_idx} but first index {first_idx} not in metadata_mapping, skipping")
                    continue
            elif source_idx is None:
                logger.warning(f"Node '{llm_node.name}' missing source_index, using first source")
                metadata = metadata_mapping[0] if 0 in metadata_mapping else {}
            else:
                logger.error(f"Node '{llm_node.name}' has invalid source_index {source_idx}, skipping")
                continue

            schema_nodes.append(self._llm_node_to_schema_node(llm_node, metadata))

        logger.info(f"Extracted {len(schema_nodes)} nodes from batch")

        # Collect book IDs for book-scoped relationships
        book_ids = set()
        for item in data_items:
            if item.metadata and 'book_id' in item.metadata:
                try:
                    book_id_value = item.metadata['book_id']
                    # Handle case where book_id might be a list
                    if isinstance(book_id_value, list):
                        if book_id_value:  # Non-empty list
                            book_ids.add(int(book_id_value[0]))
                            logger.warning(f"book_id was a list: {book_id_value}, using first element: {book_id_value[0]}")
                    else:
                        book_ids.add(int(book_id_value))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse book_id from metadata: {item.metadata.get('book_id')}, error: {e}")
                    pass

        # Generate relationships
        relationships = await self._generate_all_relationships(schema_nodes, formatted_text, book_ids)

        # Save to database
        await self._save_graph_update(schema_nodes, relationships)

    async def ingest_unstructured_data_to_graph(self, data: UnstructuredData):
        """
        Ingest single unstructured data item into the graph.
        """
        if self.graph_ops is None or self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        text = self.preprocess_data(data)

        # Extract nodes from content
        new_nodes = await self.extract_nodes(text, data.metadata or {})
        if not new_nodes:
            logger.info("No nodes generated from unstructured data")
            return

        # Collect book IDs for book-scoped relationships
        book_ids = set()
        if data.metadata and 'book_id' in data.metadata:
            try:
                book_id_value = data.metadata['book_id']
                # Handle case where book_id might be a list
                if isinstance(book_id_value, list):
                    if book_id_value:  # Non-empty list
                        book_ids.add(int(book_id_value[0]))
                        logger.warning(f"book_id was a list: {book_id_value}, using first element: {book_id_value[0]}")
                else:
                    book_ids.add(int(book_id_value))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse book_id from metadata: {data.metadata.get('book_id')}, error: {e}")
                pass

        logger.info(f"Collected book_ids: {book_ids} for relationship generation")

        # Generate relationships
        relationships = await self._generate_all_relationships(new_nodes, text, book_ids)

        # Save to database
        await self._save_graph_update(new_nodes, relationships)

    def preprocess_data(self, data: UnstructuredData) -> str:
        """
        Preprocess the data, combine relevant fields into a single string.
        """
        preprocessed = f"{data.title}\n{data.content}\n"
        if data.metadata:
            preprocessed += "\n".join([f"{k}: {v}" for k, v in data.metadata.items()])
        return preprocessed

    def preprocess_batch_data(self, data_items: List[UnstructuredData]) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Preprocess batch data with indexed source formatting.

        Args:
            data_items: List of UnstructuredData items to process

        Returns:
            Tuple of (formatted_text, metadata_mapping)
            - formatted_text: Text with indexed sources "Source [0]:", "Source [1]:", etc.
            - metadata_mapping: Dict mapping source index to original metadata
        """
        text_parts = []
        metadata_mapping = {}

        for idx, data in enumerate(data_items):
            # Format source with index
            source_text = f"Source [{idx}]:\n"
            source_text += f"Title: {data.title}\n"
            source_text += f"Content: {data.content}\n"

            # Include metadata in the text for LLM context (optional)
            if data.metadata:
                source_text += "Metadata:\n"
                for k, v in data.metadata.items():
                    source_text += f"  {k}: {v}\n"

            text_parts.append(source_text)

            # Store metadata mapping for later
            metadata_mapping[idx] = data.metadata or {}

        formatted_text = "\n\n".join(text_parts)
        return formatted_text, metadata_mapping

    async def extract_nodes(self, text: str, metadata: Dict[str, str] = {}) -> List[Node]:
        """
        Extract nodes from the unstructured text.
        Entity IDs (book_id, highlight_id, writing_id) and chunk_ids from metadata are applied to all extracted nodes.
        """

        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=[])
        llm_nodes = await get_nodes(text, graph_context)

        # Theme consolidation handled by two-tier system:
        # - Tier 1: Similarity merging (automatic during graph save)
        # - Tier 2: Frequency-based pruning (after ingestion completes)

        # Extract chunk_ids from metadata (comma-separated string to array)
        chunk_ids_from_metadata = []
        if 'chunk_ids' in metadata:
            # Can be comma-separated string: "uuid1,uuid2,uuid3"
            chunk_str = metadata['chunk_ids'].strip()
            if chunk_str:
                chunk_ids_from_metadata = [chunk.strip() for chunk in chunk_str.split(',')]

        # Extract entity IDs from metadata (convert string values to int arrays)
        book_ids = []
        if 'book_id' in metadata:
            try:
                book_ids = [int(metadata['book_id'])]
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert book_id from metadata: {e}")

        highlight_ids = []
        if 'highlight_id' in metadata:
            try:
                highlight_ids = [int(metadata['highlight_id'])]
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert highlight_id from metadata: {e}")

        writing_ids = []
        if 'writing_id' in metadata:
            try:
                writing_ids = [int(metadata['writing_id'])]
            except (ValueError, TypeError):
                pass

        # Extract date from metadata
        from datetime import datetime
        annotation_date = None
        if 'date' in metadata:
            try:
                # Parse ISO 8601 date string or accept datetime object
                date_value = metadata['date']
                if isinstance(date_value, str):
                    annotation_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                elif isinstance(date_value, datetime):
                    annotation_date = date_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date from metadata: {e}")

        # Convert LLM nodes to schema nodes, applying IDs from metadata
        nodes = []
        for node in llm_nodes:
            # Skip UserNote nodes that are empty or placeholders
            if node.type == "UserNote":
                node_name_lower = node.name.lower().strip()
                if (not node.name.strip() or
                    node_name_lower in [
                        "no user note provided", "no note", "no note provided",
                        "none", "n/a", "na", "not applicable",
                        "no annotation", "no comment", "no user comment",
                        "-", "--", "...", "null"
                    ]):
                    logger.warning(f"Skipping empty/placeholder UserNote node: '{node.name}'")
                    continue

            llm_book_id = getattr(node, 'book_id', [])
            final_book_id = llm_book_id or book_ids

            # Initialize temporal fields
            current_time = datetime.utcnow()

            nodes.append(Node(
                name=node.name,
                type=node.type,
                chunk_ids=getattr(node, 'chunk_ids', []) or chunk_ids_from_metadata,
                book_id=final_book_id,
                highlight_id=getattr(node, 'highlight_id', []) or highlight_ids,
                writing_id=getattr(node, 'writing_id', []) or writing_ids,
                discipline=getattr(node, 'discipline', ''),
                confidence=getattr(node, 'confidence', 0.0),
                created_at=annotation_date or current_time
            ))

        return nodes

    async def generate_relationships(self, nodes: List[Node], context_description: str = "") -> List[Relationship]:
        """
        Generate core relationships between nodes.
        Only creates relationships that are strongly justified.
        """
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=nodes)
        llm_nodes = self._schema_nodes_to_llm_nodes(nodes)
        llm_relationships, _ = await get_relationships(llm_nodes, graph_context)  # Ignore the ID mapping
        return self._llm_relationships_to_schema(llm_relationships)

    async def generate_cross_relationships(self, new_nodes: List[Node], existing_context: str) -> List[Relationship]:
        """
        Generate relationships between new and existing nodes.
        Only creates relationships when there's a strong, meaningful connection.
        """
        llm_nodes = self._schema_nodes_to_llm_nodes(new_nodes)
        llm_relationships, _ = await get_relationships(llm_nodes, existing_context)  # Ignore the ID mapping
        return self._llm_relationships_to_schema(llm_relationships)

    async def generate_book_scoped_relationships(self, new_nodes: List[Node], book_id: int) -> List[Relationship]:
        """
        Generate relationships between new nodes and existing nodes from the SAME book
        using vector similarity search filtered by book_id.

        This enables connections between reading progress data and highlights from the same source.

        Args:
            new_nodes: Newly extracted nodes
            book_id: Book ID to filter by

        Returns:
            List of relationships between new and existing same-book nodes
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        relationships = []

        # Generate embeddings for all new nodes
        node_texts = [node.name for node in new_nodes]
        embeddings = generate_embeddings(node_texts)

        # OPTIMIZATION: Prepare all LLM tasks for parallel execution
        async def process_single_node(node: Node, embedding) -> List[Relationship]:
            """Process a single node and return its relationships"""
            if not embedding:
                return []

            try:
                # Vector search filtered by book_id
                similar_nodes = await self.graph_ops.neo4j_manager.query_text_similarity(
                    keyword_embedding=embedding,
                    user_id=self.user_id,
                    book_id=book_id,  # Key: filter to same book only
                    limit=10  # Higher limit since we're pre-filtering by book
                )

                if not similar_nodes:
                    return []

                # Filter to meaningful similarity (lower threshold for same book)
                # Also exclude the node itself if it was already created
                # CRITICAL: Exclude CognitiveLevel nodes - they can only have HAS_UNDERSTANDING_LEVEL edges
                # CRITICAL: Theme nodes require VERY HIGH similarity (0.85+) to avoid spaghetti graph
                min_score = 0.85 if node.type == "Theme" else 0.70
                relevant_nodes = [
                    n for n in similar_nodes
                    if n.get('score', 0) >= min_score
                    and n.get('nodeName') != node.name
                    and n.get('type', '') != 'CognitiveLevel'  # CognitiveLevel nodes cannot have other relationships
                ]

                if not relevant_nodes:
                    return []

                # Format context for LLM to generate appropriate relationships
                context = self._format_nodes_for_context(relevant_nodes)

                # Use LLM to generate relationships with the same-book context
                node_relationships = await self.generate_cross_relationships([node], context)

                logger.debug(
                    f"Found {len(node_relationships)} book-scoped relationships for node '{node.name}' "
                    f"(book_id: {book_id}, similar nodes: {len(relevant_nodes)})"
                )
                return node_relationships

            except Exception as e:
                logger.warning(f"Error finding book-scoped relationships for node '{node.name}': {e}")
                return []

        # Execute all LLM calls in parallel
        import asyncio
        tasks = [process_single_node(node, embedding) for node, embedding in zip(new_nodes, embeddings)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        for result in all_results:
            if isinstance(result, Exception):
                logger.warning(f"Task failed with exception: {result}")
            elif isinstance(result, list) and result:
                relationships.extend(result)

        return relationships

    async def generate_contrast_relationships(self, new_nodes: List[Node], book_ids: set[int]) -> List[Relationship]:
        """
        Generate relationships with existing concepts by detecting both similarities AND contrasts/oppositions.

        This method specifically targets concepts that vector similarity alone would miss:
        - Philosophical opposites (e.g., "individualism" vs "collectivism")
        - Contrasting viewpoints (e.g., "free market" vs "central planning")
        - Concepts in the same domain but with different positions

        Strategy:
        1. For each new Concept node, find candidates via:
           - Mid-range vector similarity (0.40-0.69) - not too similar, not unrelated
           - OR same discipline (regardless of similarity score)
           - Filtered by book_id to stay contextually relevant
        2. Pass candidates to LLM with specialized contrast-detection prompt
        3. LLM identifies: CONTRASTS_WITH, OPPOSES, SUPPORTS, CHALLENGES, RELATED_TO

        Args:
            new_nodes: Newly extracted nodes
            book_ids: Set of book IDs to search within

        Returns:
            List of relationships including contrasting/opposing connections
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        # Only apply to Concept nodes (not Theme, Person, etc.)
        concept_nodes = [n for n in new_nodes if n.type == "Concept"]
        if not concept_nodes:
            logger.info("Phase 4: No Concept nodes to process for contrast detection")
            return []

        logger.info(f"Phase 4: Starting contrast detection for {len(concept_nodes)} Concept node(s)")
        relationships = []

        # Generate embeddings for all concept nodes
        node_texts = [node.name for node in concept_nodes]
        embeddings = generate_embeddings(node_texts)

        # OPTIMIZATION: Prepare all LLM tasks for parallel execution
        async def process_single_contrast(book_id: int, node: Node, embedding) -> List[Relationship]:
            """Process a single node for contrast detection and return relationships"""
            if not embedding:
                logger.debug(f"Phase 4: Skipping node '{node.name[:50]}...' - no embedding")
                return []

            try:
                logger.debug(f"Phase 4: Searching candidates for '{node.name[:60]}...' (book_id: {book_id})")
                # Vector search filtered by book_id
                similar_nodes = await self.graph_ops.neo4j_manager.query_text_similarity(
                    keyword_embedding=embedding,
                    user_id=self.user_id,
                    book_id=book_id,
                    limit=30  # Get more candidates for contrast detection
                )

                logger.info(f"Phase 4: Found {len(similar_nodes) if similar_nodes else 0} similar nodes for '{node.name[:60]}...'")

                if not similar_nodes:
                    return []

                # Filter candidates for contrast detection:
                # 1. Mid-range similarity (0.40-0.69) - potentially contrasting
                # 2. OR same discipline (concepts in same field often have opposing views)
                # 3. Exclude very high similarity (>= 0.70) - those are handled by book-scoped relationships
                # 4. Exclude very low similarity (< 0.40) - likely unrelated
                # 5. Exclude the node itself

                node_discipline = getattr(node, 'discipline', None) or node.properties.get('discipline') if hasattr(node, 'properties') else None

                candidate_nodes = []
                for n in similar_nodes:
                    score = n.get('score', 0.0)
                    node_name = n.get('nodeName', '')
                    candidate_discipline = n.get('discipline')
                    candidate_type = n.get('type', '')

                    # CRITICAL: Skip CognitiveLevel nodes - they can only have HAS_UNDERSTANDING_LEVEL edges
                    if candidate_type == 'CognitiveLevel':
                        continue

                    # Skip self and very high similarity (handled elsewhere)
                    if node_name == node.name or score >= 0.70:
                        continue

                    # CRITICAL: Theme nodes require VERY HIGH similarity (0.75+) for contrast detection
                    # to avoid creating too many weak relationships
                    if node.type == "Theme":
                        # For Theme nodes, only include if score is 0.60-0.70 (narrower range)
                        # and has very high semantic overlap
                        if 0.60 <= score < 0.70:
                            candidate_nodes.append(n)
                        continue  # Skip the general mid-range logic for Theme nodes

                    # Include if mid-range similarity OR related discipline (for non-Theme nodes)
                    in_mid_range = 0.40 <= score < 0.70

                    # Calculate discipline similarity (fuzzy matching)
                    related_discipline = False
                    if node_discipline and candidate_discipline and score >= 0.30:
                        related_discipline = self._disciplines_are_related(
                            node_discipline,
                            candidate_discipline
                        )

                    if in_mid_range or related_discipline:
                        candidate_nodes.append(n)

                # Limit to top 15 candidates to control LLM costs
                candidate_nodes = candidate_nodes[:15]

                logger.info(
                    f"Phase 4: After filtering, {len(candidate_nodes)} candidates remain for '{node.name[:60]}...'"
                )
                if candidate_nodes:
                    # Log sample of candidates with their scores
                    sample_candidates = candidate_nodes[:3]
                    for cand in sample_candidates:
                        logger.debug(
                            f"  Candidate: '{cand.get('nodeName', '')[:50]}...' "
                            f"(score: {cand.get('score', 0.0):.3f}, discipline: {cand.get('discipline', 'N/A')})"
                        )

                if not candidate_nodes:
                    return []

                # Format candidates for LLM
                candidates_str = self._format_candidates_for_contrast_detection(candidate_nodes)

                # Call LLM with specialized contrast detection prompt
                from persona.llm.llm_graph import detect_contrasts
                contrast_relationships = await detect_contrasts([node], candidates_str)

                logger.debug(
                    f"Found {len(contrast_relationships)} contrast/semantic relationships for '{node.name}' "
                    f"(book_id: {book_id}, candidates: {len(candidate_nodes)})"
                )
                return contrast_relationships

            except Exception as e:
                logger.warning(f"Error finding contrast relationships for node '{node.name}': {e}")
                return []

        # Execute all LLM calls in parallel (across all book_id × concept combinations)
        import asyncio
        tasks = []
        for book_id in book_ids:
            logger.info(f"Phase 4: Processing book_id {book_id}")
            for node, embedding in zip(concept_nodes, embeddings):
                tasks.append(process_single_contrast(book_id, node, embedding))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        for result in all_results:
            if isinstance(result, Exception):
                logger.warning(f"Contrast detection task failed with exception: {result}")
            elif isinstance(result, list) and result:
                relationships.extend(result)

        return relationships

    def _disciplines_are_related(self, discipline1: str, discipline2: str) -> bool:
        """
        Check if two disciplines are related using fuzzy string matching.

        This allows matching of related disciplines like:
        - "Philosophy" and "Political Philosophy"
        - "Economics" and "Behavioral Economics"
        - "Psychology" and "Cognitive Psychology"

        Args:
            discipline1: First discipline string
            discipline2: Second discipline string

        Returns:
            True if disciplines are related (similar enough), False otherwise
        """
        if not discipline1 or not discipline2:
            return False

        # Exact match
        if discipline1 == discipline2:
            return True

        # Normalize: lowercase and remove common words
        def normalize(d: str) -> set:
            """Extract significant words from discipline name"""
            d_lower = d.lower()
            # Remove common modifiers that don't change the core discipline
            stopwords = {'and', 'of', 'the', 'a', 'an', 'in'}
            words = set(d_lower.split()) - stopwords
            return words

        words1 = normalize(discipline1)
        words2 = normalize(discipline2)

        # Check for word overlap (e.g., "Philosophy" in "Political Philosophy")
        # At least one significant word must overlap
        overlap = words1 & words2
        if overlap:
            return True

        # Check if one discipline is a substring of another
        # (e.g., "Economy" in "Political Economy")
        if discipline1.lower() in discipline2.lower() or discipline2.lower() in discipline1.lower():
            return True

        # Check for common discipline families with fuzzy matching
        discipline_families = [
            {'philosophy', 'ethics', 'political philosophy', 'moral philosophy', 'metaphysics'},
            {'economics', 'political economy', 'behavioral economics', 'microeconomics', 'macroeconomics'},
            {'psychology', 'cognitive psychology', 'behavioral psychology', 'social psychology'},
            {'sociology', 'social science', 'political science'},
            {'history', 'political history', 'social history', 'economic history'},
            {'science', 'natural science', 'physical science', 'life science'},
            {'mathematics', 'statistics', 'applied mathematics'},
            {'literature', 'literary studies', 'comparative literature', 'creative writing'},
        ]

        d1_lower = discipline1.lower()
        d2_lower = discipline2.lower()

        for family in discipline_families:
            # Check if both disciplines belong to the same family
            in_family_1 = any(d in d1_lower for d in family)
            in_family_2 = any(d in d2_lower for d in family)
            if in_family_1 and in_family_2:
                return True

        return False

    def _format_candidates_for_contrast_detection(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Format candidate nodes for contrast detection prompt.

        Args:
            nodes: List of node dicts from vector search

        Returns:
            Formatted string listing candidate concepts
        """
        if not nodes:
            return ""

        context_parts = ["Candidate Concepts (may be related through similarity OR contrast):\n"]
        for i, node in enumerate(nodes, 1):
            node_name = node.get('nodeName', 'Unknown')
            score = node.get('score', 0.0)
            discipline = node.get('discipline', 'Unknown')
            context_parts.append(f"{i}. \"{node_name}\" (discipline: {discipline}, similarity: {score:.2f})")

        return "\n".join(context_parts)

    def _format_nodes_for_context(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Format a list of similar nodes into context string for LLM relationship generation.

        Args:
            nodes: List of node dicts from vector search (with 'nodeName', 'score', etc.)

        Returns:
            Formatted context string
        """
        if not nodes:
            return ""

        context_parts = ["Existing related nodes from the same source:\n"]
        for i, node in enumerate(nodes, 1):
            node_name = node.get('nodeName', 'Unknown')
            score = node.get('score', 0.0)
            context_parts.append(f"{i}. {node_name} (similarity: {score:.2f})")

        return "\n".join(context_parts)

    async def get_relevant_graph_context(self, user_id: str, nodes: List[Node], max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the given nodes.
        """
        if self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        return await self.graph_context_retriever.get_relevant_graph_context(nodes=nodes, user_id=user_id, max_hops=max_hops)

    async def close(self):
        await self.__aexit__(None, None, None)



    
