from typing import List, Optional
import tiktoken
from .client_factory import get_embedding_client
from server.logging_config import get_logger

logger = get_logger(__name__)

# Conservative token limit for embeddings (8192 is max, using 7500 for safety)
EMBEDDING_TOKEN_LIMIT = 7500


def _batch_texts_by_tokens(texts: List[str], encoding_name: str = "cl100k_base") -> List[List[int]]:
    """
    Batch texts into groups that stay under the token limit.

    Args:
        texts: List of texts to batch
        encoding_name: Tiktoken encoding name (cl100k_base is used by text-embedding-3-small)

    Returns:
        List of batches, where each batch is a list of indices into the original texts list
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}. Using character count approximation.")
        # Fallback: approximate 4 chars per token
        batches = []
        current_batch = []
        current_tokens = 0

        for i, text in enumerate(texts):
            estimated_tokens = len(text) // 4
            if estimated_tokens > 8000:
                logger.warning(f"Text at index {i} estimated at {estimated_tokens} tokens (>{8000}), truncating: {text[:100]}...")
                texts[i] = text[:32000]  # ~8000 tokens worth of characters
                estimated_tokens = 8000

            if current_tokens + estimated_tokens > EMBEDDING_TOKEN_LIMIT and current_batch:
                batches.append(current_batch)
                current_batch = [i]
                current_tokens = estimated_tokens
            else:
                current_batch.append(i)
                current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    # Count tokens for each text
    batches = []
    current_batch = []
    current_tokens = 0

    for i, text in enumerate(texts):
        token_count = len(encoding.encode(text))

        # Handle edge case: single text exceeds limit
        if token_count > 8000:
            logger.warning(f"Text at index {i} has {token_count} tokens (>8000), truncating: {text[:100]}...")
            # Truncate to first 8000 tokens
            tokens = encoding.encode(text)[:8000]
            texts[i] = encoding.decode(tokens)
            token_count = 8000

        # Start new batch if adding this text would exceed limit
        if current_tokens + token_count > EMBEDDING_TOKEN_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = [i]
            current_tokens = token_count
        else:
            current_batch.append(i)
            current_tokens += token_count

    # Add final batch
    if current_batch:
        batches.append(current_batch)

    return batches


def generate_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the configured LLM service.
    This is a synchronous function that wraps the async embedding generation.
    Implements smart batching to stay under token limits (7500 tokens per batch).

    Args:
        texts: List of texts to generate embeddings for
        model: Model name (ignored, uses configured model)

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    try:
        import asyncio

        # Batch texts by token count to stay under limits
        batches = _batch_texts_by_tokens(texts)

        if len(batches) > 1:
            logger.info(f"Batching {len(texts)} texts into {len(batches)} batches to stay under token limit")

        # Get the embedding client
        client = get_embedding_client()

        # Process each batch and collect results
        all_embeddings = [None] * len(texts)  # Preserve order

        for batch_indices in batches:
            batch_texts = [texts[i] for i in batch_indices]

            # Run the async embeddings function for this batch
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a different approach
                # This is a fallback for sync usage in async contexts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, client.embeddings(batch_texts))
                    batch_embeddings = future.result()
            else:
                batch_embeddings = asyncio.run(client.embeddings(batch_texts))

            # Place embeddings back in correct positions
            for i, embedding in zip(batch_indices, batch_embeddings):
                all_embeddings[i] = embedding

        return all_embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
        return [[]] * len(texts)  # Return a list of empty lists to maintain alignment with input texts


async def generate_embeddings_async(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Async version of generate_embeddings.
    Implements smart batching to stay under token limits (7500 tokens per batch).

    Args:
        texts: List of texts to generate embeddings for
        model: Model name (ignored, uses configured model)

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    try:
        # Batch texts by token count to stay under limits
        batches = _batch_texts_by_tokens(texts)

        if len(batches) > 1:
            logger.info(f"Batching {len(texts)} texts into {len(batches)} batches to stay under token limit")

        # Get the embedding client
        client = get_embedding_client()

        # Process each batch and collect results
        all_embeddings = [None] * len(texts)  # Preserve order

        for batch_indices in batches:
            batch_texts = [texts[i] for i in batch_indices]
            batch_embeddings = await client.embeddings(batch_texts)

            # Place embeddings back in correct positions
            for i, embedding in zip(batch_indices, batch_embeddings):
                all_embeddings[i] = embedding

        return all_embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
        return [[]] * len(texts)  # Return a list of empty lists to maintain alignment with input texts
