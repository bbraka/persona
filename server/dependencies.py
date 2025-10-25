from fastapi import Depends, Request, Path
from persona.core.graph_ops import GraphOps
from typing import Annotated
from server.logging_config import get_logger

logger = get_logger(__name__)

def get_graph_ops(request: Request) -> GraphOps:
    """Dependency to get the global GraphOps instance from app.state"""
    return request.app.state.graph_ops

# Type alias for easier usage in service methods
GraphOpsDep = Annotated[GraphOps, Depends(get_graph_ops)]

async def ensure_user_exists(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops)
) -> str:
    """
    Dependency that ensures a user exists before processing the request.
    If the user doesn't exist, it creates them automatically.

    Returns:
        str: The validated user_id
    """
    if not await graph_ops.user_exists(user_id):
        logger.info(f"Auto-creating user: {user_id}")
        await graph_ops.create_user(user_id)
        logger.debug(f"User {user_id} auto-created successfully")

    return user_id

# Type alias for endpoints that need user auto-creation
UserIdDep = Annotated[str, Depends(ensure_user_exists)] 