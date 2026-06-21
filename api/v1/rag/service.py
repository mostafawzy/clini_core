from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from core.config import get_settings
from core.llm import get_llm
from core.embeddings import get_embeddings
from core.vectorstore import get_vectorstore_path

from api.v1.rag.schemas import RAGResponse, RAGSource


class RAGService:
    def __init__(self) -> None:
        self._embeddings = None
        self._vector_store: Optional[FAISS] = None
        self._llm = None
        self._indexed_docs: list[str] = []

        settings = get_settings()

        #  CHUNKING 
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._initialised = False

        
        self.base_path = Path(get_vectorstore_path())
        self.raw_path = self.base_path / "raw"
        self.index_path = self.base_path / "indexes"

        for p in [self.raw_path, self.index_path]:
            p.mkdir(parents=True, exist_ok=True)

    
    def _ensure_init(self) -> None:
        if self._initialised:
            return

        logger.info("Initializing RAG service...")

        self._embeddings = get_embeddings()
        self._llm = get_llm()

        #  LOAD EXISTING INDEX 
        index_file = self.index_path / "index.faiss"

        if index_file.exists():
            logger.info(f"Loading FAISS index from {self.index_path}")

            self._vector_store = FAISS.load_local(
                str(self.index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )

            meta_path = self.index_path / "doc_list.json"
            if meta_path.exists():
                self._indexed_docs = json.loads(meta_path.read_text())

        else:
            logger.info("No existing FAISS index found")

        self._initialised = True

    #  INGEST PDF 
    async def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        self._ensure_init()

        logger.info(f"Ingesting file: {filename}")

        doc_id = str(uuid.uuid4())

        #  SAVE RAW FILE 
        raw_file = self.raw_path / f"{doc_id}_{filename}"
        raw_file.write_bytes(pdf_bytes)

        #  LOAD PDF 
        loader = PyPDFLoader(str(raw_file))
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source": filename,
                "doc_id": doc_id,
                "doc_type": "clinical_pdf",
                "stage": "raw",
            })

        #  SPLIT 
        chunks: list[Document] = self._splitter.split_documents(pages)

        if not chunks:
            raise ValueError("No text extracted from PDF")

        #  CHUNKS WITH METADATA
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source": filename,
                "stage": "embedded",
                "chunk_index": i,
            })
            enriched_chunks.append(chunk)

        assert self._embeddings is not None

        #  STORE 
        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(
                enriched_chunks,
                self._embeddings
            )
        else:
            self._vector_store.add_documents(enriched_chunks)

        self._persist()

        #  TRACK DOCS 
        if filename not in self._indexed_docs:
            self._indexed_docs.append(filename)
            self._save_doc_list()

        return {
            "doc_id": doc_id,
            "chunks": len(enriched_chunks)
        }

    #  QUERY 
    async def query(self, question: str, top_k: int = 5) -> RAGResponse:
        self._ensure_init()

        if self._vector_store is None:
            return RAGResponse(
                query=question,
                answer="No documents indexed yet.",
                sources=[],
            )

        if self._llm is None:
            raise ValueError("LLM is not initialized")

        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        prompt = ChatPromptTemplate.from_template("""
You are Clinicore AI, a clinical knowledge assistant.

Use only the provided context.
If the answer is not in the context, say so clearly.

Context:
{context}

Question:
{question}

Answer:
""")

        def format_docs(docs):
            return "\n\n".join(
                f"[SOURCE:{d.metadata.get('source')}]\n{d.page_content}"
                for d in docs
            )

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        answer = await rag_chain.ainvoke(question)
        docs = await retriever.ainvoke(question)

        # -------- SOURCES --------
        seen = set()
        sources = []

        for doc in docs:
            snippet = doc.page_content[:120]

            if snippet in seen:
                continue

            seen.add(snippet)

            sources.append(
                RAGSource(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                )
            )

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
        )

    #  LIST DOCS 
    async def list_documents(self) -> dict:
        self._ensure_init()
        return {
            "documents": self._indexed_docs,
            "count": len(self._indexed_docs),
        }

    #  PERSIST 
    def _persist(self) -> None:
        if self._vector_store is None:
            return
        self._vector_store.save_local(str(self.index_path))

    def _save_doc_list(self) -> None:
        (self.index_path / "doc_list.json").write_text(
            json.dumps(self._indexed_docs, indent=2)
        )


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service

    if _rag_service is None:
        _rag_service = RAGService()

    return _rag_service