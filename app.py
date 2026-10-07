"""DATA 폴더의 문서로 질문에 답하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def get_openai_api_key() -> str:
    """Streamlit Cloud Secrets를 우선 사용하고, 로컬에서는 .env를 사용합니다."""
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        secret_key = None

    return str(secret_key or os.getenv("OPENAI_API_KEY") or "").strip()


def load_pdf_documents(data_dir: Path) -> list[Document]:
    """DATA 폴더 아래의 모든 PDF를 페이지 단위 문서로 읽습니다."""
    documents: list[Document] = []

    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise RuntimeError("DATA 폴더에서 읽을 수 있는 PDF 문서를 찾지 못했습니다.")
    return documents


@st.cache_resource(show_spinner=False)
def build_retriever() -> Any:
    """문서를 분할하고 OpenAI 임베딩으로 InMemoryVectorStore를 만듭니다."""
    # .env의 API 키는 앱 시작 시 한 번만 읽고, 키 자체는 화면에 표시하지 않습니다.
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

    pages = load_pdf_documents(DATA_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "다. ", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        # 청크 크기를 직접 제한했으므로 별도의 tiktoken/transformers 다운로드를 생략합니다.
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def build_answer_chain() -> Any:
    """최신 LCEL 파이프라인으로 답변 생성 체인을 구성합니다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 제공된 문서만 근거로 답하는 문서 기반 챗봇입니다.
문서에 명시된 내용만 사용하고, 문서에 없거나 확실하지 않은 내용은 추측하지 마세요.
답을 확인할 수 없으면 정확히 '문서에서 확인할 수 없습니다.'라고 답하세요.
답변은 자연스러운 한글 띄어쓰기와 문장부호를 사용하세요.
첫 문장에는 결론을 먼저 쓰고, 설명할 내용이 여러 개면 항목별 Markdown 목록으로 정리하세요.
문단 사이에는 한 줄을 띄워 읽기 쉽게 작성하세요.

문서 내용:
{context}""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)
    return prompt | model | StrOutputParser()


def format_answer(answer: str) -> str:
    """모델 답변의 불필요한 공백과 과도한 빈 줄을 정리합니다."""
    cleaned = answer.strip()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def make_context(documents: list[Document]) -> str:
    """검색된 문서를 모델이 출처와 함께 구분해서 읽도록 만듭니다."""
    parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "알 수 없는 파일")
        page = document.metadata.get("page", "?")
        parts.append(f"[문서 {index} | 파일: {source} | 페이지: {page}]\n{document.page_content}")
    return "\n\n".join(parts)


def evidence_sentence(text: str, question: str) -> str:
    """검색 결과에서 질문과 관련성이 높은 한 문장을 골라 표시합니다."""
    normalized = " ".join(text.split())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", normalized) if part.strip()]
    if not sentences:
        return normalized[:400]

    question_words = {word for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower())}
    scored = []
    for position, sentence in enumerate(sentences):
        sentence_words = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", sentence.lower()))
        scored.append((len(question_words & sentence_words), -position, sentence))
    best = max(scored, key=lambda item: (item[0], item[1]))[2]
    return best[:400]


def render_sources(documents: list[Document], question: str) -> None:
    """답변 아래에 실제 검색된 파일명, 페이지, 근거 문장을 표시합니다."""
    shown: set[tuple[str, int]] = set()
    source_items: list[tuple[str, int, str]] = []
    for document in documents:
        source = str(document.metadata.get("source", "알 수 없는 파일"))
        page = int(document.metadata.get("page", 0))
        key = (source, page)
        if key in shown:
            continue
        shown.add(key)
        evidence = evidence_sentence(document.page_content, question)
        source_items.append((source, page, evidence))

    with st.expander(f"출처 및 근거 ({len(source_items)}개)", expanded=False):
        for source, page, evidence in source_items:
            # PDF의 특수문자가 Markdown/LaTeX로 해석되지 않도록 안전하게 표시합니다.
            safe_evidence = evidence.replace("$", "\\$").replace("`", "\\`")
            st.markdown(f"**{source}** · p. {page}")
            st.markdown(f"> {safe_evidence}")


def main() -> None:
    """Streamlit 화면과 질문 처리 흐름을 실행합니다."""
    load_dotenv(PROJECT_ROOT / ".env")

    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] p { line-height: 1.75; }
        [data-testid="stExpander"] p { line-height: 1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더의 문서에 있는 내용만 근거로 답변합니다.")

    if not get_openai_api_key():
        st.warning("프로젝트 루트의 .env 파일에 OPENAI_API_KEY를 입력해주세요.")
        st.stop()

    try:
        retriever = build_retriever()
        answer_chain = build_answer_chain()
    except Exception as error:  # 사용자에게 실행에 필요한 문제를 설명합니다.
        st.error(f"문서 검색 환경을 준비하지 못했습니다: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"], message["question"])

    question = st.chat_input("문서에 대해 질문하세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 작성하는 중..."):
            documents = retriever.invoke(question)
            context = make_context(documents)
            answer = format_answer(answer_chain.invoke({"context": context, "question": question}))
        st.markdown(answer)
        render_sources(documents, question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": documents,
            "question": question,
        }
    )


if __name__ == "__main__":
    main()
