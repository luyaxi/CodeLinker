"""Configuration module.

This module should be loaded after the environment variables are set,
and before any other modules are loaded.

The module initializes the global configuration.
"""


import os
from typing import Literal
from pydantic import BaseModel
from copy import deepcopy

class CodeLinkerConfig(BaseModel):
    """Module configuration model."""
    class APIConfig(BaseModel):
        api_key: str
        model: str
        class Config:
            allow_arbitrary_types = True
            extra = "allow"
            
    api_keys: dict[str, list[APIConfig]] = {}
    
    max_retry_times: int = 3
    
    class RequestConfig(BaseModel):
        format: Literal[
            "chat",
            "tool_call",
            "function_call"] = "tool_call"
        json_mode: bool = False
        schema_validation: bool = True
        dynamic_json_fix: bool = True

        default_request_lib: str = 'openai'
        default_completions_model: str = "gpt-3.5-turbo-16k"
        default_embeding_model: str = "text-embedding-ada-002"
        default_timeout: int = 600
        
        req_name_mapping: dict[str, str] = {
            "gpt-4v": "gpt-4-vision",
            "gpt4": "gpt-4",
            "gpt4-32": "gpt-4-32k",
            "gpt-35-16k": "gpt-3.5-turbo-16k",
        }

        use_cache: bool = False
        save_completions: bool = False
        # default to save in current working directory 
        save_completions_path: str = os.path.join(os.getcwd(),"cache","completions")

    request: RequestConfig = RequestConfig()
    
    class ExectionConfig(BaseModel):
        max_ai_functions_tokens: int = 12800
        max_message_tokens: int = 15872

    execution: ExectionConfig = ExectionConfig()

    class VisionConfig(BaseModel):
        class ImageModalConfig(BaseModel):
            enable: bool = False
            default_model: str = "gpt-4-vision"
        image: ImageModalConfig = ImageModalConfig()

    multimodal: VisionConfig = VisionConfig()
    
    def to_dict(self):
        return self.model_dump(exclude=['api_keys', 'databases'])
    
    @classmethod
    def from_toml(cls, toml_stream: str| os.PathLike) -> "CodeLinkerConfig":
        import toml
        if os.path.exists(toml_stream):
            with open(toml_stream, 'r') as f:
                toml_stream = f.read()
        config = toml.loads(toml_stream)
        return CodeLinkerConfig(**config)
    
    def get_model_name(self,model_name:str=None) -> str:
        if model_name is None:
            model_name = self.request.default_completions_model
            return model_name

        return self.request.req_name_mapping.get(model_name, model_name)
    
    def get_apiconfig_by_model(self, model_name: str = None) -> "CodeLinkerConfig.APIConfig":
        """
        Get API configuration for a model by its name.
        Return default model if the given model name is not found.

        The function first normalizes the name, then fetches the API keys for this model
        from the CONFIG and rotates the keys.

        Args:
            model_name (str): Name of the model.

        Returns:
            dict: Dictionary containing the fetched API configuration.
        """
        normalized_model_name = self.get_model_name(model_name)
        apiconfig = deepcopy(self.api_keys[normalized_model_name][0])
        self.api_keys[normalized_model_name].append(
            self.api_keys[normalized_model_name].pop(0))
        return apiconfig
