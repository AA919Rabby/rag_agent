import google.generativeai as genai
from app.config import settings
from app.services.rag_service import rag_service
import logging


logger=logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model=genai.GenerativeModel("gemini-1.5-flash")
    def generative_response(self,prompt:str):
        try:
            context,sources=rag_service.query_vector_store(prompt)
            augmented_prompt=f"""
            You are an expert document analysis assistant. Answer the  user's question accurately using ONLY the context provided below.
            If the answer cannot be found in the  context, reply with: "I cannot find answer in the uploaded documents."
            Context: {context}
            User Question: {prompt}
            """
            response=self.model.generate_content(augmented_prompt)
            return {
                "answer":response.text,
                "source_documents":list(set(sources))
            }
        except Exception as e:
            logger.error(f"GEMINI API Error: {str(e)}")
            raise RuntimeError(
                f"Failed to  generate response: {str(e)}"
            )

gemini_service=GeminiService()


