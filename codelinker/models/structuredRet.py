from pydantic import BaseModel,Field
from typing import Any

class StructureSchema(BaseModel):
    name: str
    description: str = ""
    parameters: dict

class StructuredRet(BaseModel):
    name: str = Field(description="Name of the structured return")
    content: Any = Field(description="Content of the structured return")
