"""RAG 서비스 (LLM 전용)"""
from pathlib import Path
from typing import List
from app.services.vector_store import vector_store_service
from app.services.pdf_processor import pdf_processor
from app.models.llm_manager import llm_manager
from app.schemas.api_schemas import AnswerResponse
from app.core.config import settings
from app.core.logging import log
import time

class RAGService:
    """RAG 서비스 클래스 (LLM 전용)"""
    
    def __init__(self):
        self.vector_store = vector_store_service
        self.pdf_processor = pdf_processor
        self.llm = llm_manager
    
    def answer_question(self, question: str) -> AnswerResponse:
        """질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            AnswerResponse 객체
        """
        log.info(f"Answering question: '{question}'")
        start_time = time.time()
        
        try:
            # 1. 유사 문서 검색
            retrieved_docs = self.vector_store.search(
                question, 
                k=settings.TOP_K
            )
            
            if not retrieved_docs:
                log.warning("No relevant documents found")
                return AnswerResponse(
                    question=question,
                    answer="관련 정보를 찾을 수 없습니다.",
                    confidence=0.0,
                    sources=[]
                )
            
            # 2. 컨텍스트 구성
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. 프롬프트 생성
            prompt = self._create_prompt(question, context)
            
            # 4. LLM으로 답변 생성
            answer = self.llm.generate_answer(prompt)
            
            # 5. 출처 추출
            sources = list(set([
                doc.metadata.get("source", "unknown") 
                for doc in retrieved_docs
            ]))
            
            # 6. 신뢰도 계산
            confidence = self._calculate_confidence(retrieved_docs, answer)
            
            elapsed = time.time() - start_time
            log.info(f"Generated answer: '{answer}' (confidence: {confidence:.2f}, time: {elapsed:.2f}s)")
            
            return AnswerResponse(
                question=question,
                answer=answer,
                confidence=round(confidence, 2),
                sources=sources
            )
            
        except Exception as e:
            log.error(f"Failed to answer question: {e}")
            raise
    
    def add_pdf_document(self, pdf_path: Path) -> int:
        """PDF 문서를 벡터 DB에 추가
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            생성된 청크 수
        """
        log.info(f"Adding PDF document: {pdf_path.name}")
        
        try:
            # PDF 처리
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # 벡터 스토어에 추가
            self.vector_store.add_documents(chunks)
            
            log.info(f"Successfully added {len(chunks)} chunks from {pdf_path.name}")
            return len(chunks)
            
        except Exception as e:
            log.error(f"Failed to add PDF document: {e}")
            raise
    
    def _create_prompt(self, question: str, context: str) -> str:
        """프롬프트 생성 (값 포함하도록 개선)"""
        
        # 컨텍스트 충분히 제공
        short_context = context[:600] if len(context) > 600 else context
        
        # 값을 반드시 포함하도록 프롬프트 수정
        prompt = f"""문서:
    {short_context}

    질문: {question}

    규칙:
    1. 문서에서 질문의 답을 찾아 **구체적인 값과 단위**를 포함하여 답하세요
    2. 문서에 없는 내용은 절대 만들지 마세요
    3. 답변 형식 예시:
    - "HBM 수출액: 620억 달러"
    - "NVIDIA 비중: 65%"
    - "총 수출액: 980억 달러"
    4. 항목만 쓰지 말고 반드시 값을 함께 작성하세요

    답변:"""
        
        return prompt

    
    def _calculate_confidence(self, documents: List, answer: str) -> float:
        """신뢰도 계산
        
        Args:
            documents: 검색된 문서 리스트
            answer: 생성된 답변
            
        Returns:
            신뢰도 (0-1)
        """
        # 기본 신뢰도
        base_confidence = 0.75
        
        # 문서 수에 따른 보너스
        doc_bonus = min(len(documents) * 0.05, 0.15)
        
        # 답변 길이 체크 (너무 짧거나 긴 경우 감소)
        if len(answer) < 10:
            length_penalty = -0.1
        elif len(answer) > 200:
            length_penalty = -0.05
        else:
            length_penalty = 0
        
        # 특정 키워드 포함 여부 (숫자, 금액 등)
        keyword_bonus = 0
        if any(char.isdigit() for char in answer):
            keyword_bonus += 0.05
        if any(word in answer for word in ['억', '달러', '원', '%']):
            keyword_bonus += 0.05
        
        total_confidence = base_confidence + doc_bonus + length_penalty + keyword_bonus
        
        return min(0.95, max(0.5, total_confidence))

# 싱글톤 인스턴스
rag_service = RAGService()
