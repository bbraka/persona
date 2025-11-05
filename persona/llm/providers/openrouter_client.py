"""
OpenRouter LLM client implementation.
OpenRouter provides a unified API for accessing multiple LLM providers.
"""

import openai
from typing import List, Dict, Any, Optional
from .base import BaseLLMClient, ChatMessage, ChatResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class OpenRouterClient(BaseLLMClient):
    """OpenRouter LLM client - uses OpenAI-compatible API"""

    def __init__(self, api_key: str, chat_model: str, **kwargs):
        super().__init__(model_name=chat_model, **kwargs)
        self.api_key = api_key
        self.chat_model = chat_model

        # Initialize OpenAI-compatible client with OpenRouter base URL
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using OpenRouter API"""
        try:
            # Convert our standard message format to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Prepare request parameters
            request_params = {
                "model": self.chat_model,
                "messages": openai_messages,
            }

            # Only include temperature if explicitly provided (for models that support it)
            if temperature is not None:
                request_params["temperature"] = temperature

            if max_tokens:
                request_params["max_tokens"] = max_tokens

            if response_format:
                request_params["response_format"] = response_format

            # Add any additional parameters
            request_params.update(kwargs)

            # Make the API call
            response = await self.async_client.chat.completions.create(**request_params)

            return ChatResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None
            )

        except openai.APIStatusError as e:
            error_msg = f"OpenRouter chat error: {e}"

            # Log rate limit headers if available
            if e.response is not None:
                headers = e.response.headers
                rate_limit_info = {
                    'limit_requests': headers.get('x-ratelimit-limit-requests'),
                    'limit_tokens': headers.get('x-ratelimit-limit-tokens'),
                    'remaining_requests': headers.get('x-ratelimit-remaining-requests'),
                    'remaining_tokens': headers.get('x-ratelimit-remaining-tokens'),
                    'reset_requests': headers.get('x-ratelimit-reset-requests'),
                    'reset_tokens': headers.get('x-ratelimit-reset-tokens'),
                }
                # Filter out None values
                rate_limit_info = {k: v for k, v in rate_limit_info.items() if v is not None}
                if rate_limit_info:
                    error_msg += f"\nRate limit info: {rate_limit_info}"
                else:
                    # If no rate limit headers, log all headers for debugging
                    error_msg += f"\nResponse headers: {dict(headers)}"

            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error(f"OpenRouter chat error: {e}")
            raise

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        OpenRouter does not support embeddings.
        This method will raise an error if called.
        """
        raise NotImplementedError(
            "OpenRouter does not support embeddings. "
            "Please use OpenAI or Azure OpenAI for EMBEDDING_SERVICE."
        )

    def get_provider_name(self) -> str:
        return "openrouter"

    def supports_json_mode(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return False
