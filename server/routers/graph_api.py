from typing import Optional, Union, List
from fastapi import APIRouter, HTTPException, status, Path, Depends, Body, Response, Query
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.models.schema import NodeModel, RelationshipModel, GraphUpdateModel
from persona.core.constructor import GraphConstructor
from persona.llm.prompts import sample_statements, ASTRONAUT_PROMPT, SPACE_SCHOOL_CHAT
from persona.models.schema import UnstructuredData
from persona.core.constructor import GraphContextRetriever
from persona.core.rag_interface import RAGInterface
from persona.models.schema import UserCreate, RAGQuery, RAGResponse
from persona.services.user_service import UserService
from persona.services.ingest_service import IngestService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.services.custom_data_service import CustomDataService
from persona.services.graph_ui_service import GraphUIService
from persona.models.schema import LearnRequest, LearnResponse, AskRequest, AskResponse, GraphSchema, CustomGraphUpdate, CustomNodeData, CustomRelationshipData, GraphUIDataResponse
from server.dependencies import get_graph_ops, ensure_user_exists
from server.logging_config import get_logger
import re

logger = get_logger(__name__)


router = APIRouter()

# Regex for validating user IDs. Allows alphanumeric chars, hyphens, and underscores.
# This provides a basic level of sanitization to prevent injection or invalid characters.
USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")

def is_valid_user_id(user_id: str) -> bool:
    """Check if the user ID matches the allowed pattern."""
    return bool(USER_ID_REGEX.match(user_id))

@router.get("/version")
def get_version():
    return {"version": "1.0.0"}

@router.post("/users/{user_id}")
async def create_user(
    response: Response,
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not is_valid_user_id(user_id):
            raise ValueError("Invalid user ID format.")

        logger.info(f"Creating user: {user_id}")
        result = await UserService.create_user(user_id, graph_ops)
        
        # Set appropriate status code based on whether user was created or already existed
        if result["status"] == "exists":
            response.status_code = 200  # OK - user already exists
            logger.debug(f"User {user_id} already exists")
        else:
            response.status_code = 201  # Created - new user
            logger.info(f"User {user_id} created successfully")
            
        return result
            
    except ValueError as e:
        logger.warning(f"Invalid user ID format: {user_id} - {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid user ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while creating user")

@router.delete("/users/{user_id}", status_code=200, description="Delete an existing user from the system")
async def delete_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not is_valid_user_id(user_id):
            raise ValueError("Invalid user ID format.")

        logger.info(f"Deleting user: {user_id}")
        
        # Check if user exists first
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Attempted to delete non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        await UserService.delete_user(user_id, graph_ops)
        logger.info(f"User {user_id} deleted successfully")
        return {"message": f"User {user_id} deleted successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid user ID provided for deletion: {user_id} - {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid user ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while deleting user")

@router.post("/users/{user_id}/ingest", status_code=201)
async def ingest_data(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    data: UnstructuredData = Body(
        ...,
        openapi_examples={
            "example": {
                "summary": "Single item ingestion",
                "description": "Ingest a single highlight, note, or writing",
                "value": {
                    "title": "Book Highlight",
                    "content": "Professional jealousy can lead to betrayal",
                    "metadata": {
                        "book_id": "36",
                        "highlight_id": "84104",
                        "date": "2025-10-31T08:48:01.906Z"
                    }
                }
            }
        }
    ),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    """
    Ingest a single unstructured data item into the graph.

    For batch ingestion of multiple items, use the `/users/{user_id}/ingest/batch` endpoint instead.

    ## Use Cases
    - Single highlight from a book
    - Individual note or writing
    - One-off data ingestion

    ## For Multiple Items
    Use the batch endpoint (`/users/{user_id}/ingest/batch`) which provides:
    - Single LLM call (90% cost reduction)
    - Accurate metadata mapping via source indexing
    - Cross-source relationship detection
    """
    try:
        logger.info(f"Ingesting single data item for user: {user_id}")

        # Validate data content
        if not data.content or len(data.content.strip()) == 0:
            logger.warning(f"Empty content provided for user {user_id}")
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        result = await IngestService.ingest_data(user_id, data, graph_ops)
        logger.info(f"Data ingested successfully for user {user_id}")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to ingest data for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while ingesting data")

@router.post("/users/{user_id}/ingest/batch", status_code=201)
async def ingest_batch(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    data: List[UnstructuredData] = Body(
        ...,
        openapi_examples={
            "example": {
                "summary": "Batch ingestion with source indexing",
                "description": "Ingest multiple items in one request with accurate metadata mapping",
                "value": [
                    {
                        "title": "Book Highlight Id: 84104",
                        "content": "Professional jealousy can lead to betrayal",
                        "metadata": {
                            "book_id": "36",
                            "highlight_id": "84104",
                            "date": "2025-10-31T08:48:01.906Z"
                        }
                    },
                    {
                        "title": "Book Highlight Id: 84105",
                        "content": "Hope sustains people through suffering",
                        "metadata": {
                            "book_id": "36",
                            "highlight_id": "84105",
                            "date": "2025-10-31T08:49:15.123Z"
                        }
                    },
                    {
                        "title": "Book Highlight Id: 84106",
                        "content": "Isolation transforms personality over time",
                        "metadata": {
                            "book_id": "36",
                            "highlight_id": "84106",
                            "date": "2025-10-31T08:50:32.456Z"
                        }
                    }
                ]
            }
        }
    ),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    """
    Ingest multiple unstructured data items in a single batch request.

    ## Batch Ingestion Benefits
    - **90% cost reduction**: Single LLM call instead of N calls
    - **Accurate metadata mapping**: Source indexing ensures each node gets correct metadata
    - **Cross-source relationships**: Detects concepts spanning multiple sources
    - **Atomic operation**: All items processed together or transaction rolls back

    ## How Source Indexing Works
    1. Each item is formatted as "Source [0]:", "Source [1]:", etc.
    2. LLM extracts nodes with `source_index` field
    3. Nodes are mapped back to original metadata via index
    4. Each node gets its specific book_id, highlight_id, writing_id

    ## Best Practices
    - Batch size: 3-50 items for optimal performance
    - All items should be related (e.g., highlights from same reading session)
    - Include all metadata for accurate tracking
    """
    try:
        # Validate batch
        if not data:
            logger.warning(f"Empty batch provided for user {user_id}")
            raise HTTPException(status_code=400, detail="Batch cannot be empty")

        # Validate each item in batch
        for idx, item in enumerate(data):
            if not item.content or len(item.content.strip()) == 0:
                logger.warning(f"Empty content in batch item {idx} for user {user_id}")
                raise HTTPException(status_code=400, detail=f"Content cannot be empty in batch item {idx}")

        logger.info(f"Ingesting batch of {len(data)} items for user: {user_id}")
        result = await IngestService.ingest_data(user_id, data, graph_ops)
        logger.info(f"Batch ingestion completed successfully for user {user_id}")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to ingest batch for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while ingesting batch")

@router.post("/users/{user_id}/rag/query", response_model=RAGResponse)
async def rag_query(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    query: RAGQuery = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            logger.warning(f"Empty query received for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Validate query length
        if len(query.query.strip()) > 1000:
            logger.warning(f"Query too long for user {user_id}: {len(query.query)} characters")
            raise HTTPException(status_code=400, detail="Query is too long (max 1000 characters)")
            
        logger.info(f"Processing RAG query for user {user_id}: {query.query[:100]}...")
        result = await RAGService.query(user_id, query.query, graph_ops)
        logger.info(f"RAG query completed successfully for user {user_id}")
        return RAGResponse(answer=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG query for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please ensure Neo4j is running and accessible."
            )
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing query")

@router.post("/users/{user_id}/rag/query-vector", status_code=status.HTTP_200_OK)
async def rag_query_vector(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    query: RAGQuery = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            logger.warning(f"Empty vector query received for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")

        logger.info(f"Processing vector RAG query for user {user_id}: {query.query[:100]}...")
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        response = await rag.query_vector_only(query.query)
        logger.info(f"Vector RAG query completed successfully for user {user_id}")
        return {"query": query.query, "response": response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during vector-only RAG query for user {user_id}: {e}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing vector query")

@router.post("/users/{user_id}/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_insights(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    ask_request: AskRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not ask_request:
            logger.warning(f"Empty ask request received for user {user_id}")
            raise HTTPException(status_code=400, detail="Request body is required")
            
        if not ask_request.query or len(ask_request.query.strip()) == 0:
            logger.warning(f"Empty query in ask request for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")

        logger.info(f"Processing ask insights for user {user_id}: {ask_request.query[:100]}...")
        response = await AskService.ask_insights(user_id, ask_request, graph_ops)
        logger.info(f"Ask insights completed successfully for user {user_id}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid ask request format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in ask insights for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing insights request")

@router.post("/users/{user_id}/custom-data", status_code=status.HTTP_200_OK)
async def update_custom_data(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    update: CustomGraphUpdate = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    """
    Update or create custom structured data in the graph
    """
    try:
        if not update:
            logger.warning(f"Empty custom data update received for user {user_id}")
            raise HTTPException(status_code=400, detail="Request body is required")

        # Validate update content
        if not update.nodes and not update.relationships:
            logger.warning(f"Empty custom data update for user {user_id}")
            raise HTTPException(status_code=400, detail="At least one node or relationship must be provided")
            
        logger.info(f"Processing custom data update for user {user_id}: {len(update.nodes or [])} nodes, {len(update.relationships or [])} relationships")
        custom_service = CustomDataService(graph_ops)
        result = await custom_service.update_custom_data(user_id, update)
        logger.info(f"Custom data update completed successfully for user {user_id}")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid custom data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in custom data update for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while updating custom data")

@router.get("/users/{user_id}/graph-ui-data", response_model=GraphUIDataResponse, status_code=status.HTTP_200_OK)
async def get_graph_ui_data(
    user_id: str = Depends(ensure_user_exists),  # Auto-creates user if doesn't exist
    book_id: Optional[int] = Query(None, description="Filter nodes by book ID"),
    highlight_id: Optional[int] = Query(None, description="Filter nodes by highlight ID"),
    writing_id: Optional[int] = Query(None, description="Filter nodes by writing ID"),
    date_from: Optional[str] = Query(None, description="Filter nodes created on or after this date (ISO 8601 format, e.g., '2025-02-24')"),
    date_to: Optional[str] = Query(None, description="Filter nodes created on or before this date (ISO 8601 format, e.g., '2025-02-26')"),
    limit: Optional[int] = Query(None, description="Maximum number of nodes to return (for pagination). Uses efficient cursor-based pagination with composite index."),
    cursor: Optional[str] = Query(None, description="Cursor for pagination (node name to start after). Get this from pagination.next_cursor in previous response."),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    """
    Retrieve comprehensive graph data for UI visualization with optional filtering, pagination, and cursor support.

    Query Parameters:
    - book_id: Filter nodes that belong to this book ID
    - highlight_id: Filter nodes that belong to this highlight ID
    - writing_id: Filter nodes that belong to this writing ID
    - date_from: Filter nodes created on or after this date (ISO 8601, e.g., "2025-02-24")
    - date_to: Filter nodes created on or before this date (ISO 8601, e.g., "2025-02-26")
    - limit: Maximum number of nodes to return (None = all nodes). Example: ?limit=100
    - cursor: Cursor for next page (from pagination.next_cursor). Example: ?limit=100&cursor=Milton+Friedman

    Pagination:
    - Uses efficient cursor-based pagination with composite index on (UserId, name)
    - First request: ?limit=100 (returns first 100 nodes ordered by name)
    - Next requests: ?limit=100&cursor={next_cursor from previous response}
    - Response includes pagination metadata: {total_nodes, returned_nodes, has_more, next_cursor, limit}

    Returns:
    - topics: Aggregated by discipline with entity counts, relationship counts, and Bloom level distribution
    - insights: High-confidence nodes (confidence >= 0.7)
    - sources: Aggregated entity ID statistics
    - nodes: Filtered graph nodes with properties (paginated if limit specified)
    - relationships: Graph relationships connecting returned nodes only
    - pagination: Pagination metadata (only present if limit specified)
    """
    try:
        if not is_valid_user_id(user_id):
            raise ValueError("Invalid user ID format.")

        logger.info(f"Fetching graph UI data for user {user_id} with filters - book_id: {book_id}, highlight_id: {highlight_id}, writing_id: {writing_id}, date_from: {date_from}, date_to: {date_to}, limit: {limit}, cursor: {cursor}")
        result = await GraphUIService.get_graph_ui_data(
            user_id=user_id,
            graph_ops=graph_ops,
            book_id=book_id,
            highlight_id=highlight_id,
            writing_id=writing_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            cursor=cursor
        )
        logger.info(f"Graph UI data fetched successfully for user {user_id}")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid user ID format for graph UI data: {user_id} - {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid user ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching graph UI data for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while fetching graph UI data")