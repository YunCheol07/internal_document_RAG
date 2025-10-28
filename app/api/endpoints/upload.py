"""PDF 업로드 엔드포인트"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.api_schemas import UploadResponse
from app.services.rag_service import rag_service
from app.core.config import settings
from app.core.logging import log
import shutil

router = APIRouter()

@router.post("/upload-pdf", response_model=UploadResponse, tags=["Upload"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF 파일 업로드 및 벡터 DB 추가
    
    - **file**: 업로드할 PDF 파일 (multipart/form-data)
    
    Returns:
        UploadResponse: 업로드 결과 정보
    """
    log.info(f"Received file upload request: {file.filename}")
    
    # 파일 확장자 검증
    if not file.filename.endswith('.pdf'):
        log.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail="PDF 파일만 업로드 가능합니다."
        )
    
    try:
        # 파일 저장
        pdf_path = settings.PDF_DOCUMENTS_PATH / file.filename
        
        log.info(f"Saving file to: {pdf_path}")
        
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 벡터 DB에 추가
        chunks_created = rag_service.add_pdf_document(pdf_path)
        
        log.info(f"Successfully processed {file.filename}")
        
        return UploadResponse(
            message="PDF 파일이 성공적으로 업로드되었습니다.",
            filename=file.filename,
            chunks_created=chunks_created,
            status="success"
        )
    
    except ValueError as e:
        log.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )
    
    except Exception as e:
        log.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"파일 처리 중 오류 발생: {str(e)}"
        )
