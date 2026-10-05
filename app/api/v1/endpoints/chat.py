from fastapi import APIRouter,File,status,HTTPException,UploadFile
from app.models.schemas import ChatRequest,ChatResponse,UploadResponse
from app.services.rag_service import rag_service
from app.services.gemini_service import gemini_service


router=APIRouter()
@router.post("/upload",response_model=UploadResponse,status_code=status.HTTP_201_CREATED)
async def upload_document(file:UploadFile=File(...)):
    if not file.filename.endswith((".pdf",".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and text files are supported"
        )
    try:
        filename=await rag_service.save_and_process_file(file)
        return UploadResponse(
            filename=filename,
            message="FIle uploaded, parsed, and  successfully indexed into  ChromaDB!"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/post-chat",response_model=ChatResponse,status_code=status.HTTP_200_OK)
def rag_chat_endpoint(payload:ChatRequest):
    try:
        result=gemini_service.generative_response(payload.prompt)
        return ChatResponse(
            answer=result["answer"],
            source_documents=result["source_documents"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


