"""벡터 스토어 관리 서비스"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
from pathlib import Path
import shutil
from app.core.config import settings
from app.core.logging import log

class VectorStoreService:
    """벡터 스토어 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        log.info("Initializing VectorStoreService")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """새로운 벡터 스토어 생성
        
        Args:
            documents: Document 객체 리스트
            
        Returns:
            FAISS 벡터 스토어
        """
        log.info(f"Creating new vectorstore with {len(documents)} documents")
        
        try:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            log.info("Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            log.error(f"Failed to create vectorstore: {e}")
            raise
    
    def save_vectorstore(self, vectorstore: FAISS):
        """벡터 스토어 저장
        
        Args:
            vectorstore: 저장할 FAISS 벡터 스토어
        """
        try:
            vectorstore.save_local(str(settings.VECTOR_DB_PATH))
            log.info(f"Vectorstore saved to {settings.VECTOR_DB_PATH}")
            
        except Exception as e:
            log.error(f"Failed to save vectorstore: {e}")
            raise
    
    def load_vectorstore(self) -> FAISS:
        """기존 벡터 스토어 로드
        
        Returns:
            FAISS 벡터 스토어
        """
        index_file = settings.VECTOR_DB_PATH / "index.faiss"
        
        if not index_file.exists():
            raise FileNotFoundError(
                f"Vectorstore index file not found at {index_file}"
            )
        
        log.info(f"Loading vectorstore from {settings.VECTOR_DB_PATH}")
        
        try:
            vectorstore = FAISS.load_local(
                str(settings.VECTOR_DB_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.vectorstore = vectorstore
            log.info("Vectorstore loaded successfully")
            
            return vectorstore
            
        except Exception as e:
            log.error(f"Failed to load vectorstore: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> int:
        """기존 벡터 스토어에 문서 추가
        
        Args:
            documents: 추가할 Document 객체 리스트
            
        Returns:
            추가된 문서 수
        """
        log.info(f"Adding {len(documents)} documents to vectorstore")
        
        try:
            # 벡터 스토어 존재 여부 확인
            vectorstore_exists = (
                settings.VECTOR_DB_PATH.exists() and 
                (settings.VECTOR_DB_PATH / "index.faiss").exists()
            )
            
            if not vectorstore_exists:
                # 벡터 스토어가 없으면 새로 생성
                log.info("Vectorstore not found, creating new one")
                self.vectorstore = self.create_vectorstore(documents)
                self.save_vectorstore(self.vectorstore)
                return len(documents)
            
            # 기존 벡터 스토어 로드
            if self.vectorstore is None:
                self.vectorstore = self.load_vectorstore()
            
            # 문서 추가
            self.vectorstore.add_documents(documents)
            self.save_vectorstore(self.vectorstore)
            
            log.info(f"Successfully added {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            log.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """유사 문서 검색 (관련성 향상)"""
        if k is None:
            k = settings.TOP_K
        
        log.info(f"Searching for: '{query}' (top {k})")
        
        try:
            if self.vectorstore is None:
                self.vectorstore = self.load_vectorstore()
            
            # 유사도 검색 (score_threshold 추가 가능)
            results = self.vectorstore.similarity_search(query, k=k)
            
            # 결과 필터링: 너무 관련성 낮은 문서 제거
            filtered_results = []
            for doc in results:
                # 키워드 매칭 체크
                query_keywords = set(query.lower().split())
                doc_keywords = set(doc.page_content.lower().split())
                
                # 최소 1개 키워드는 일치해야 함
                if query_keywords & doc_keywords:
                    filtered_results.append(doc)
            
            # 필터링된 결과가 없으면 원본 반환
            final_results = filtered_results if filtered_results else results
            
            log.info(f"Found {len(final_results)} relevant results")
            return final_results
            
        except Exception as e:
            log.error(f"Search failed: {e}")
            raise

    
    def reset_vectorstore(self):
        """벡터 스토어 초기화"""
        log.warning("Resetting vectorstore")
        
        if settings.VECTOR_DB_PATH.exists():
            shutil.rmtree(settings.VECTOR_DB_PATH)
            settings.VECTOR_DB_PATH.mkdir(exist_ok=True)
        
        self.vectorstore = None
        log.info("Vectorstore reset complete")
    
    def get_stats(self) -> dict:
        """벡터 스토어 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        if self.vectorstore is None:
            try:
                self.vectorstore = self.load_vectorstore()
            except FileNotFoundError:
                return {"total_documents": 0, "status": "not_initialized"}
        
        return {
            "total_documents": self.vectorstore.index.ntotal,
            "status": "loaded"
        }

# 싱글톤 인스턴스
vector_store_service = VectorStoreService()

