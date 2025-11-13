import json
from typing import List, Optional, Tuple, Dict, Any
from persona.llm.prompts import GET_NODES, GET_RELATIONSHIPS, DETECT_CONTRASTS, GENERATE_COMMUNITIES, GENERATE_STRUCTURED_INSIGHTS, ASSESS_COGNITIVE_LEVEL
from persona.models.schema import EntityExtractionResponse, NodesAndRelationshipsResponse, CommunityStructure, AskResponse, AskRequest, create_dynamic_schema
from pydantic import BaseModel, Field, field_validator, model_validator
from persona.utils.instructions_reader import INSTRUCTIONS
from server.logging_config import get_logger
from server.config import config
from .client_factory import get_chat_client
from .providers.base import ChatMessage

logger = get_logger(__name__)

class Node(BaseModel):
    model_config = {"extra": "ignore"}  # Ignore extra fields from LLM responses

    name: str = Field(..., description="The node content - can be a simple label (e.g., 'Techno Music') or a narrative fragment (e.g., 'Deeply moved by classical music in empty spaces')")
    type: str = Field(..., description="The type/category of the node (e.g., 'Identity', 'Belief', 'Preference', 'Goal', 'Event', 'Relationship', etc.)")
    source_index: Optional[int | List[int]] = Field(None, description="Index of source(s) this node came from in batch processing (e.g., 0, 1, or [0, 2])")
    chunk_ids: Optional[List[str]] = Field(default_factory=list, description="Array of chunk IDs linking this node to multiple source sections")
    book_id: Optional[List[int]] = Field(default_factory=list, description="Array of book IDs associated with this node")
    highlight_id: Optional[List[int]] = Field(default_factory=list, description="Array of highlight IDs associated with this node")
    writing_id: Optional[List[int]] = Field(default_factory=list, description="Array of writing IDs associated with this node")
    discipline: Optional[str] = Field(None, description="Academic/knowledge domain of the node")
    confidence: Optional[float] = Field(None, description="Extraction quality score (0.0-1.0)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties of the node")

    @model_validator(mode='after')
    def pack_properties(self) -> 'Node':
        """Pack discipline and confidence into properties dict.
        NOTE: chunk_ids are NOT packed into properties - they remain top-level."""
        # Ensure properties dict is initialized
        if self.properties is None:
            self.properties = {}

        # Pack fields into properties
        if self.discipline is not None and 'discipline' not in self.properties:
            self.properties['discipline'] = self.discipline
        if self.confidence is not None and 'confidence' not in self.properties:
            self.properties['confidence'] = self.confidence

        return self

class Relationship(BaseModel):
    source: str
    relation: str
    target: str

class RelationshipWithID(BaseModel):
    source_id: str = Field(..., description="Temporary ID of the source node (e.g., 'Node1')")
    relation: str = Field(..., description="Type of relationship")
    target_id: str = Field(..., description="Temporary ID of the target node (e.g., 'Node2')")

class GraphResponse(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes in the graph")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships between nodes")

async def get_nodes(text: str, graph_context: str) -> List[Node]:
    """
    Extract nodes from provided text using the configured LLM service.
    Returns nodes that can be either simple labels or narrative fragments.
    """
    try:
        combined_instructions = f"App Objective: {INSTRUCTIONS}\n\nExisting Graph Context: {graph_context}\n\nNode Extraction Task: {GET_NODES}"
        
        messages = [
            ChatMessage(role="system", content=combined_instructions),
            ChatMessage(role="user", content=text)
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs: Dict[str, Any] = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)

        # Parse JSON response
        json_data = json.loads(response.content)
        
        # Validate and convert to Node objects
        if "nodes" in json_data:
            nodes = [Node(**node_data) for node_data in json_data["nodes"]]
            return nodes
        else:
            logger.warning("No 'nodes' key found in LLM response")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in get_nodes: {e}")
        return []
    except Exception as e:
        logger.error(f"Error while extracting nodes: {e}")
        return []

async def get_relationships(nodes: List[Node], graph_context: str) -> Tuple[List[Relationship], Dict[str, str]]:
    """
    Generate relationships based on the list of nodes and existing graph context using the configured LLM service.
    Returns a tuple of (relationships, id_mapping) where id_mapping maps temporary IDs to node names.
    """
    if not nodes:
        return [], {}
    
    # Create temporary ID mapping
    id_mapping = {}
    nodes_with_ids = []
    
    for i, node in enumerate(nodes):
        temp_id = f"Node{i+1}"
        id_mapping[temp_id] = node.name
        nodes_with_ids.append(f'{temp_id}: "{node.name}"')
    
    # Format nodes for the prompt with temporary IDs
    nodes_str = '\n'.join(nodes_with_ids)
    combined_instructions = f"App Objective: {INSTRUCTIONS}\n\nRelationships Generation Task: {GET_RELATIONSHIPS}"
    
    try:
        messages = [
            ChatMessage(role="system", content=combined_instructions),
            ChatMessage(role="user", content=f"Nodes:\n{nodes_str}\n\nExisting Graph Context:\n{graph_context}")
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)
        
        # Parse JSON response
        json_data = json.loads(response.content)
        
        # Validate and convert to RelationshipWithID objects
        relationships_with_ids = []
        if "relationships" in json_data:
            for rel_data in json_data["relationships"]:
                relationships_with_ids.append(RelationshipWithID(**rel_data))
        
        # Convert RelationshipWithID back to Relationship using the mapping
        converted_relationships = []
        for rel_with_id in relationships_with_ids:
            source_name = id_mapping.get(rel_with_id.source_id)
            target_name = id_mapping.get(rel_with_id.target_id)
            
            if source_name and target_name:
                converted_relationships.append(Relationship(
                    source=source_name,
                    relation=rel_with_id.relation,
                    target=target_name
                ))
            else:
                logger.warning(f"Invalid relationship with IDs: {rel_with_id.source_id} -> {rel_with_id.target_id}")
        
        return converted_relationships, id_mapping
        
    except Exception as e:
        logger.error(f"Error while generating relationships: {e}")
        return [], {}

async def assess_cognitive_level(user_note_text: str) -> str:
    """
    Assess cognitive level based on user's note/reaction text.

    This function takes a UserNote text and determines what cognitive level
    (from Bloom's Taxonomy) the user demonstrated through their reaction.

    Args:
        user_note_text: The text of the user's note/reaction

    Returns:
        One of: "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"

    Example:
        >>> await assess_cognitive_level("interesting")
        "Remember"
        >>> await assess_cognitive_level("Rand's issue is exactly being extreme!")
        "Evaluate"
    """
    if not user_note_text or not user_note_text.strip():
        logger.warning("Empty UserNote text provided for cognitive assessment, defaulting to Remember")
        return "Remember"

    try:
        messages = [
            ChatMessage(role="system", content=ASSESS_COGNITIVE_LEVEL),
            ChatMessage(role="user", content=f"UserNote text: {user_note_text}")
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)

        # Parse JSON response
        json_data = json.loads(response.content)

        cognitive_level = json_data.get("cognitive_level", "Remember")

        # Validate the response
        valid_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        if cognitive_level not in valid_levels:
            logger.warning(f"Invalid cognitive level '{cognitive_level}' returned, defaulting to Remember")
            return "Remember"

        logger.debug(f"Assessed cognitive level for note '{user_note_text[:50]}...': {cognitive_level}")
        return cognitive_level

    except Exception as e:
        logger.error(f"Error while assessing cognitive level: {e}")
        return "Remember"

async def detect_contrasts(new_nodes: List[Node], candidates_context: str) -> List[Relationship]:
    """
    Detect semantic relationships (including contrasts/oppositions) between new concepts and existing candidates.

    This function uses a specialized prompt to find:
    - Contrasting/opposing concepts (CONTRASTS_WITH, OPPOSES, CHALLENGES)
    - Supporting concepts (SUPPORTS, SIMILAR_TO, REINFORCES)
    - Other semantic relationships (RELATED_TO, APPLIES_TO, etc.)

    Args:
        new_nodes: List of new Node objects to analyze
        candidates_context: Formatted string of candidate concepts from vector search

    Returns:
        List of Relationship objects
    """
    if not new_nodes:
        return []

    # Format new concepts for the prompt
    new_concepts_str = '\n'.join([f'- "{node.name}"' for node in new_nodes])

    combined_prompt = f"""
{DETECT_CONTRASTS}

**New Concepts**:
{new_concepts_str}

{candidates_context}

Please analyze these concepts and return ALL meaningful relationships in JSON format.
"""

    try:
        messages = [
            ChatMessage(role="system", content=f"App Objective: {INSTRUCTIONS}"),
            ChatMessage(role="user", content=combined_prompt)
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)

        # Parse JSON response
        json_data = json.loads(response.content)

        # Validate and convert to Relationship objects
        relationships = []
        if "relationships" in json_data:
            for rel_data in json_data["relationships"]:
                try:
                    relationships.append(Relationship(**rel_data))
                except Exception as e:
                    logger.warning(f"Failed to parse relationship: {rel_data}, error: {e}")

        logger.info(f"Detected {len(relationships)} contrast/semantic relationships for {len(new_nodes)} new concepts")
        return relationships

    except Exception as e:
        logger.error(f"Error while detecting contrasts: {e}")
        return []

async def generate_response_with_context(query: str, context: str) -> str:
    """Generate a response based on query and context using the configured LLM service."""
    prompt = f"""
    Given the following context from a knowledge graph and a query, provide a detailed answer:

    Context:
    {context}

    Query: {query}

    Please provide a comprehensive answer based on the given context:
    """

    try:
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant that answers queries about a user based on the provided context from their graph."),
            ChatMessage(role="user", content=prompt)
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs: Dict[str, Any] = {"messages": messages}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating response with context: {e}")
        return "I apologize, but I encountered an error while processing your request."

async def detect_communities(subgraphs_text: str) -> CommunityStructure:
    """
    Use LLM to detect communities in the graph and organize them into headers/subheaders
    """
    try:
        messages = [
            ChatMessage(role="system", content=GENERATE_COMMUNITIES),
            ChatMessage(role="user", content=subgraphs_text)
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)
        
        # Parse JSON response
        json_data = json.loads(response.content)
        
        # Validate and convert to CommunityStructure
        community_structure = CommunityStructure(**json_data)
        logger.debug(f"Community Detection Response: {community_structure}")
        return community_structure
        
    except Exception as e:
        logger.error(f"Error in community detection: {str(e)}")
        return CommunityStructure(communityHeaders=[])

async def generate_structured_insights(ask_request: AskRequest, context: str) -> Dict[str, Any]:
    """
    Generate structured insights based on the provided context and query using the configured LLM service.
    Returns a message indicating no relevant context was found if context is empty.
    """
    # Check if context is empty or only whitespace
    if not context or not context.strip() or context.strip() == "# Relevant Graph Context":
        logger.warning(f"No relevant context found for query: {ask_request.query}")

        # Return error response in the expected schema format
        error_response = {}
        for key, value in ask_request.output_schema.items():
            if isinstance(value, list):
                error_response[key] = []
            elif isinstance(value, dict):
                error_response[key] = {"error": "No relevant context found in the knowledge graph"}
            else:
                error_response[key] = ""

        return error_response

    prompt = f"""
    IMPORTANT! DO NOT USE ANY INFORMATION OUTSIDE OF THE PROVIDED CONTEXT TO ANSWER THE QUERY.

    Based on this context from the knowledge graph:
    {context}

    Answer this query about the user: {ask_request.query}

    Provide your response following the example structure:
    {json.dumps(ask_request.output_schema, indent=2)}
    """

    logger.debug(f"Structured insights prompt: {prompt}")

    try:
        messages = [
            ChatMessage(role="system", content=GENERATE_STRUCTURED_INSIGHTS),
            ChatMessage(role="user", content=prompt)
        ]

        client = get_chat_client()

        # Build kwargs, only include temperature if configured
        kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
        if config.MACHINE_LEARNING.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.MACHINE_LEARNING.LLM_TEMPERATURE

        response = await client.chat(**kwargs)

        return json.loads(response.content)

    except Exception as e:
        logger.error(f"Error in generate_structured_insights: {e}")
        return {k: [] if isinstance(v, list) else {} for k, v in ask_request.output_schema.items()}


def validate_theme_ratio(nodes: List[Node], max_ratio: float = 0.10) -> List[Node]:
    """
    Enforce Theme:Concept ratio constraint to prevent theme over-extraction.

    This function removes the lowest-confidence Theme nodes if the ratio exceeds
    the maximum threshold (default 10% as specified in prompt guidelines).

    Args:
        nodes: List of Node objects from LLM extraction
        max_ratio: Maximum allowed ratio of Themes to Concepts (default 0.10 = 10%)

    Returns:
        Filtered list of nodes with excess Themes removed

    Example:
        - Input: 5 Concepts, 3 Themes (60% ratio)
        - Output: 5 Concepts, 0 Themes (0% ratio - removed all themes, none met criteria)
        - Log: "Theme ratio exceeded: 60%. Removing 3 themes"
    """
    themes = [n for n in nodes if n.type == "Theme"]
    concepts = [n for n in nodes if n.type == "Concept"]

    # If no concepts, remove ALL themes (themes need concepts to ground them)
    if not concepts:
        if themes:
            theme_names = [t.name[:50] + "..." if len(t.name) > 50 else t.name for t in themes]
            logger.warning(
                f"Removing {len(themes)} theme(s) - no concepts extracted. "
                f"Themes removed: {theme_names}"
            )
        return [n for n in nodes if n.type != "Theme"]

    # Calculate maximum allowed themes (at least 1 if we have 10+ concepts)
    max_themes = max(0, int(len(concepts) * max_ratio))

    # If within threshold, return unchanged
    if len(themes) <= max_themes:
        actual_ratio = (len(themes) / len(concepts) * 100) if concepts else 0
        if themes:
            logger.info(
                f"Theme extraction within limits: {len(themes)} themes / {len(concepts)} concepts = "
                f"{actual_ratio:.1f}% (target: <{max_ratio*100:.0f}%)"
            )
        return nodes

    # Ratio exceeded - need to remove themes
    # Sort themes by confidence (keep highest confidence themes)
    themes_with_confidence = []
    for theme in themes:
        confidence = 0.5  # default
        if hasattr(theme, 'confidence') and theme.confidence is not None:
            confidence = theme.confidence
        elif theme.properties and 'confidence' in theme.properties:
            confidence = theme.properties['confidence']
        themes_with_confidence.append((theme, confidence))

    # Sort by confidence descending
    themes_with_confidence.sort(key=lambda x: x[1], reverse=True)

    # Keep only max_themes highest confidence themes
    themes_to_keep = [t for t, c in themes_with_confidence[:max_themes]]
    themes_to_remove = [t for t, c in themes_with_confidence[max_themes:]]

    actual_ratio = (len(themes) / len(concepts) * 100)
    target_ratio = (max_themes / len(concepts) * 100) if concepts else 0

    removed_names = [
        (t.name[:50] + "..." if len(t.name) > 50 else t.name)
        for t in themes_to_remove
    ]

    logger.warning(
        f"Theme ratio exceeded: {len(themes)}/{len(concepts)} = {actual_ratio:.0f}% "
        f"(target: <{max_ratio*100:.0f}%). "
        f"Removing {len(themes_to_remove)} theme(s) to reach {target_ratio:.0f}%. "
        f"Removed: {removed_names}"
    )

    # Return all non-theme nodes + validated themes
    return [n for n in nodes if n.type != "Theme" or n in themes_to_keep]