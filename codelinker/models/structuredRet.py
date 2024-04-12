from pydantic import BaseModel,Field


class StructureSchema(BaseModel):
    name: str
    description: str = ""
    parameters: dict

class StructuredRet(BaseModel):
    name: str = Field(description="Name of the structured return")
    content: dict = Field(description="Content of the structured return")
