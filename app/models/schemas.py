from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    prompt:str

class ChatResponse(BaseModel):
    answer:str
    source_documents:List[str]=[]

class UploadResponse(BaseModel):
    filename:str
    message:str