from asyncio import CancelledError

from openai import AuthenticationError, PermissionDeniedError, BadRequestError, AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from ..config import CodeLinkerConfig

RETRY_ERRORS = (AuthenticationError, PermissionDeniedError,
                BadRequestError, AssertionError, CancelledError)


async def chatcompletion_request(*, config: CodeLinkerConfig, **kwargs):
    """Handle operation of OpenAI v1.x.x chat completion.

    This function operates OpenAI v1.x.x chat completion with provided
    arguments. It gets the model name, applies a JSON web token, if the
    response indicates the context length has been exceeded, it attempts
    to get a higher-capacity language model if it exists in the configuration
    and reattempts the operation. Otherwise, it will raise an error message.

    Args:
        max_lenght_fallback: If True, fallback to a longer model if the context length is exceeded.
        **kwargs: Variable length argument list including (model:str, etc.).

    Returns:
        response (dict): A dictionary containing the response from the Chat API.
        The structure of the dictionary is based on the API response format.

    Raises:
        BadRequestError: If any error occurs during chat completion operation or
        context length limit exceeded and no fallback models available.
    """
    model_name = config.get_model_name(kwargs.pop("model", None))
    chatcompletion_kwargs = config.get_apiconfig_by_model(model_name)

    request_timeout = kwargs.pop("request_timeout", None)
    if hasattr(chatcompletion_kwargs, "azure_endpoint"):
        azure_endpoint = getattr(chatcompletion_kwargs, "azure_endpoint", None)
        api_version = getattr(chatcompletion_kwargs, "api_version", None)
        api_key = getattr(chatcompletion_kwargs, "api_key", None)
        organization = getattr(chatcompletion_kwargs, "organization", None)

        client = AsyncAzureOpenAI(
            api_key=api_key,
            organization=organization,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=request_timeout
        )
    else:
        if hasattr(chatcompletion_kwargs, "base_url"):
            base_url = chatcompletion_kwargs.base_url
        elif hasattr(chatcompletion_kwargs, "api_base"):
            base_url = chatcompletion_kwargs.api_base
        else:
            base_url = None
        api_key = chatcompletion_kwargs.api_key
        organization = getattr(chatcompletion_kwargs, "organization", None)

        client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=request_timeout
        )
    chatcompletion_kwargs = chatcompletion_kwargs.model_dump(
        mode="json")
    chatcompletion_kwargs.update(kwargs)
    for k in ["azure_endpoint", "api_version", "api_key", "organization", "base_url", "api_base", "timeout"]:
        chatcompletion_kwargs.pop(k, None)

    completions: ChatCompletion = await client.chat.completions.create(**chatcompletion_kwargs)
    response = completions.model_dump()
    if response["choices"][0]["finish_reason"] == "length":
        raise BadRequestError(
            message="maximum context length exceeded", response=None, body=None
        )

    return response


async def embedding_request(config: CodeLinkerConfig, text: str, **kwargs):
    model_name = config.get_model_name(kwargs.pop("model", config.request.default_embeding_model))
    req_kwargs = config.get_apiconfig_by_model(model_name)
    
    request_timeout = kwargs.pop("request_timeout", None)
    if hasattr(req_kwargs, "azure_endpoint"):
        azure_endpoint = getattr(req_kwargs, "azure_endpoint", None)
        api_version = getattr(req_kwargs, "api_version", None)
        api_key = getattr(req_kwargs, "api_key", None)
        organization = getattr(req_kwargs, "organization", None)

        client = AsyncAzureOpenAI(
            api_key=api_key,
            organization=organization,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=request_timeout
        )
    else:
        if hasattr(req_kwargs, "base_url"):
            base_url = req_kwargs.base_url
        elif hasattr(req_kwargs, "api_base"):
            base_url = req_kwargs.api_base
        else:
            base_url = None
        api_key = req_kwargs.api_key
        organization = getattr(req_kwargs, organization, None)

        client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=request_timeout
        )
    req_kwargs = req_kwargs.model_dump(mode="json")
    req_kwargs.update(kwargs)
    for k in ["azure_endpoint", "api_version", "api_key", "organization", "base_url", "api_base", "timeout"]:
        req_kwargs.pop(k, None)
        
    response = await client.embeddings.create(
        input=[text],
        **req_kwargs
    )
    embedding = response.data[0].embedding

    return embedding
