from pydantic import TypeAdapter
from dataclasses import dataclass, field

@dataclass
class SmartFuncLabel:
    name: str
    description: str
    
    return_type: TypeAdapter
    schema: dict
    prompt: str
    
    completions_kwargs: dict = field(default_factory=dict)