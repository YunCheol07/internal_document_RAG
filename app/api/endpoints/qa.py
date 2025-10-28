"""질의응답 엔드포인트"""
from fastapi import APIRouter, HTTPException
from app.schemas.api_schemas import QuestionRequest, AnswerResponse, DetailedAnswerResponse
from app.services.rag_service import rag_service
from app.core.logging import log

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse, tags=["QA"])
async def ask_question(request: QuestionRequest):
    """
    질문에 대한 답변 생성 (answer만 반환)
    
    - **question**: 사내 문서에 대한 질문
    
    Returns:
        AnswerResponse: 답변만 포함된 JSON 응답
    
    Example Response:
        ```
        {
            "answer": "총 반도체 수출액: 980억 달러"
        }
        ```
    """
    log.info(f"Received question: '{request.question}'")
    
    try:
        # RAG 서비스로 답변 생성
        detailed_response = rag_service.answer_question(request.question)
        
        # answer만 반환
        simple_response = AnswerResponse(
            answer=detailed_response.answer
        )
        
        log.info(f"Successfully generated answer")
        return simple_response
    
    except FileNotFoundError as e:
        log.error(f"Vectorstore not found: {e}")
        raise HTTPException(
            status_code=404,
            detail="벡터 DB가 없습니다. 먼저 PDF 파일을 업로드하세요."
        )
    
    except Exception as e:
        log.error(f"Answer generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류 발생: {str(e)}"
        )

# 선택사항: 상세 정보가 필요한 경우를 위한 추가 엔드포인트
@router.post("/ask/detailed", response_model=DetailedAnswerResponse, tags=["QA"])
async def ask_question_detailed(request: QuestionRequest):
    """
    질문에 대한 상세 답변 생성 (모든 정보 포함)
    
    Returns:
        DetailedAnswerResponse: 질문, 답변, 신뢰도, 출처 포함
    """
    log.info(f"Received detailed question: '{request.question}'")
    
    try:
        response = rag_service.answer_question(request.question)
        log.info(f"Successfully generated detailed answer")
        return response
    
    except FileNotFoundError as e:
        log.error(f"Vectorstore not found: {e}")
        raise HTTPException(
            status_code=404,
            detail="벡터 DB가 없습니다."
        )
    
    except Exception as e:
        log.error(f"Answer generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류 발생: {str(e)}"
        )
