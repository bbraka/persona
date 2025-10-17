"""
Schema and Pydantic Models for the Graph Library Ops."
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UnstructuredData(BaseModel):
    title: str
    content: str
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = {}

class Node(BaseModel):
    name: str = Field(..., description="The node content - can be a simple label (e.g., 'Techno Music') or a narrative fragment (e.g., 'Deeply moved by classical music in empty spaces')")
    type: str = Field(..., description="The type/category of the node (e.g., 'Identity', 'Belief', 'Preference', 'Goal', 'Event', 'Relationship', etc.)")
    chunk_id: Optional[str] = Field(None, description="Optional chunk ID linking this node to a specific book chapter section")
    discipline: Optional[str] = Field(None, description="The discipline/category of the node (e.g., 'Music', 'Art', 'Technology', etc.)")
    bloom_level: Optional[str] = Field(None, description="The cognitive level of the node based on Bloom's taxonomy (e.g., 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create')")
    confidence: Optional[float] = Field(None, description="Confidence score for the node extraction (0.0 to 1.0)")

class Relationship(BaseModel):
    source: str
    target: str
    relation: str

class NodeModel(BaseModel):
    name: str = Field(..., description="The node content - can be a simple label or narrative fragment")
    type: Optional[str] = Field(None, description="The type/category of the node (e.g., 'Identity', 'Belief', 'Preference', etc.)")
    chunk_id: Optional[str] = Field(None, description="Optional chunk ID for tracking node source")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for the node, if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Finds peace in early morning solitude",
                "type": "Preference",
                "chunk_id": "chapter1_section3",
                "properties": {"discipline": "Lifestyle", "bloom_level": "Understand", "confidence": 0.85},
                "embedding": [0.1, 0.2, 0.3]
            }
        }


class RelationshipModel(BaseModel):
    source: str
    target: str
    relation: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "source": "Quantum Computing",
                "target": "AI",
                "relation": "RELATED_TO",
                "properties": {
                    "strength": 0.9,
                    "created_at": "2025-10-17T08:00:00Z"
                }
            }
        }



class GraphUpdateModel(BaseModel):
    nodes: List[NodeModel]
    relationships: List[RelationshipModel]

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {"name": "Node1", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, 0.3]},
                    {"name": "Node2", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, 0.3]}
                ],
                "relationships": [
                    {"source": "Node1", "target": "Node2", "relation": "CONNECTED_TO"}
                ]
            }
        }


class EntityExtractionResponse(BaseModel):
    entities: List[str] = Field(..., examples=[["Blockchain", "Quantum Computing", "Indie Games", "Sustainable Farming", "Virtual Reality"]])


class NodesAndRelationshipsResponse(BaseModel):
    nodes: List[NodeModel] = Field(
        ...,
        json_schema_extra={
            "examples": [[
                {"name": "Finds peace in early morning solitude"},
                {"name": "Techno Music"},
                {"name": "Values deep conversations"},
                {"name": "Real Madrid"},
                {"name": "Anxious about future of AI"}
            ]]
        }
    )
    relationships: List[RelationshipModel] = Field(
        ...,
        json_schema_extra={
            "examples": [[
                {"source": "Finds peace in early morning solitude", "relation": "CONTRASTS_WITH", "target": "Anxious about future of AI"},
                {"source": "Techno Music", "relation": "ENHANCES", "target": "Finds peace in early morning solitude"},
                {"source": "Values deep conversations", "relation": "REFLECTS", "target": "Anxious about future of AI"}
            ]]
        }
    )

class UserCreate(BaseModel):
    user_id: str

class RAGQuery(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str

class Subgraph(BaseModel):
    id: int
    nodes: List[str]
    relationships: List[Dict[str, str]]
    size: int
    central_nodes: List[str]  # nodes with highest degree/influence

class CommunitySubheader(BaseModel):
    subheader: str
    subgraph_ids: List[int]

class CommunityHeader(BaseModel):
    header: str
    subheaders: List[CommunitySubheader]

class CommunityStructure(BaseModel):
    communityHeaders: List[CommunityHeader]

# BYOA - Learn Anything Ask Anything Personalize Anything

class GraphSchema(BaseModel):
    name: str
    description: str
    attributes: List[str]
    relationships: List[str]
    is_seed: bool = False
    created_at: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Gaming Preferences",
                "description": "Schema for tracking user gaming preferences",
                "attributes": ["CURRENT_GAME", "FAVORITE_GENRE", "PLAYTIME"],
                "relationships": ["PLAYS", "PREFERS", "COMPLETED"],
                "is_seed": False
            }
        }

class LearnRequest(BaseModel):
    user_id: str
    graph_schema: GraphSchema
    description: str

class LearnResponse(BaseModel):
    status: str
    schema_id: str
    details: str

class AskRequest(BaseModel):
    query: str
    output_schema: Dict[str, Any] = Field(..., description="Expected output structure with example values")

class AskResponse(BaseModel):
    result: Dict[str, Any]

class AskResponseInstructor(BaseModel):
    result: Dict[str, Any]


def create_dynamic_schema(output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a JSON schema based on the provided output schema for OpenAI structured output
    """
    def create_property_schema(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            properties = {
                k: create_property_schema(v) for k, v in value.items()
            }
            return {
                "type": "object",
                "properties": properties,
                "required": list(value.keys())
            }
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                return {
                    "type": "array",
                    "items": create_property_schema(value[0])
                }
            return {
                "type": "array",
                "items": {"type": "string"}
            }
        return {"type": "string"}

    schema = {
        "type": "object",
        "properties": {
            k: create_property_schema(v) for k, v in output_schema.items()
        },
        "required": list(output_schema.keys())
    }
    
    return schema

class CustomNodeData(BaseModel):
    name: str
    perspective: Optional[str] = None
    type: Optional[str] = None
    chunk_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for the node, if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Quantum Computing",
                "type": "Concept",
                "chunk_id": "book_27_chapter_1",
                "properties": {
                    "discipline": "Computer Science",
                    "bloom_level": "Understand",
                    "confidence": "0.9"
                },
                "embedding": [0.1, 0.2, 0.3]
            }
        }

class CustomRelationshipData(BaseModel):
    source: str
    target: str
    relation_type: str
    data: Dict[str, Any] = Field(default_factory=dict)

class CustomGraphUpdate(BaseModel):
    nodes: List[CustomNodeData]
    relationships: List[CustomRelationshipData]

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "name": "Favorite Movie",
                        "properties": {"genre": "Sci-Fi", "rating": "9/10"},
                        "perspective": "user preference"
                    }
                ],
                "relationships": [
                    {
                        "source": "Favorite Movie",
                        "target": "Interstellar",
                        "relation_type": "IS",
                        "data": {"confidence": 0.9}
                    }
                ]
            }
        }

class GraphUIDataResponse(BaseModel):
    topics: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]