"""
PDF RAG Engine - Loads Employee Handbook PDF, creates vector store,
and answers questions using DeepSeek LLM via LangChain.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────

PDF_PATH = Path(__file__).parent / "Employee Handbook.pdf"
VECTOR_STORE_DIR = Path(__file__).parent / ".vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


# ─── Simple DeepSeek LLM call ─────────────────────────────────────

def call_deepseek(prompt: str, api_key: str) -> str:
    """Call DeepSeek API with a prompt and return the text response."""
    import requests
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def make_deepseek_llm(api_key: str):
    """Create a callable LLM function for LangChain."""
    def llm_func(prompt_input) -> str:
        if hasattr(prompt_input, "to_string"):
            text = prompt_input.to_string()
        elif hasattr(prompt_input, "messages"):
            text = "\n".join(m.content for m in prompt_input.messages)
        else:
            text = str(prompt_input)
        return call_deepseek(text, api_key)
    return RunnableLambda(llm_func)


# ─── Mock LLM for testing ─────────────────────────────────────────

def make_mock_llm():
    """Return a mock LLM for testing without API key."""
    def mock_func(prompt_input) -> str:
        if hasattr(prompt_input, "to_string"):
            text = prompt_input.to_string()
        elif hasattr(prompt_input, "messages"):
            text = "\n".join(m.content for m in prompt_input.messages)
        else:
            text = str(prompt_input)
        lines = text.split("\n")
        question_line = [l for l in lines if "用户的问题" in l]
        question = question_line[0].split("：", 1)[-1].strip() if question_line else "unknown"
        return (
            f"[Mock Answer] 关于「{question}」的测试回复。\n\n"
            f"请设置 DEEPSEEK_API_KEY 环境变量来获取真实回答。"
        )
    return RunnableLambda(mock_func)


# ─── PDF Loading & Chunking ───────────────────────────────────────

def load_pdf() -> list[Document]:
    """Load PDF and split into chunks."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"📖 Loading PDF: {PDF_PATH.name} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")
    loader = PyMuPDFLoader(str(PDF_PATH))
    documents = loader.load()
    print(f"   -> {len(documents)} pages loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> {len(chunks)} chunks created")
    return chunks


# ─── Vector Store ─────────────────────────────────────────────────

def get_or_create_vectorstore() -> FAISS:
    """Load existing vector store or create from PDF."""
    if VECTOR_STORE_DIR.exists():
        print(f"📦 Loading cached vector store ({VECTOR_STORE_DIR})")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("🏗️  Creating new vector store from PDF...")
    chunks = load_pdf()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    print(f"✅ Vector store saved to {VECTOR_STORE_DIR}")
    return vectorstore


# ─── QA Chain ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是公司政策助手。请根据以下提供的员工手册内容，准确回答用户的问题。

规则：
1. 如果手册中有明确答案，请直接引用并给出清晰回答
2. 如果手册中没有相关信息，请如实说"手册中没有找到相关信息"
3. 不要编造或猜测政策内容
4. 用中文回答
5. 如果涉及具体数字（天数、金额等），务必准确引用

参考内容：
{context}

请回答用户的问题："""


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectorstore, llm_func):
    """Build the RAG QA chain using LangChain's LCEL syntax."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "input": RunnablePassthrough()}
        | prompt
        | llm_func
        | StrOutputParser()
    )
    return chain


# ─── Initialize (lazy singleton) ──────────────────────────────────

_qa_chain = None


def get_qa_chain():
    """Get or initialize the QA chain (singleton)."""
    global _qa_chain

    if _qa_chain is not None:
        return _qa_chain

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    if not api_key or api_key == "sk-your-key-here":
        print("⚠️  DEEPSEEK_API_KEY not set. Using mock mode for testing.")
        llm_func = make_mock_llm()
    else:
        print("🤖 Initializing DeepSeek...")
        llm_func = make_deepseek_llm(api_key)

    print("📚 Loading knowledge base...")
    vectorstore = get_or_create_vectorstore()

    _qa_chain = build_qa_chain(vectorstore, llm_func)
    print("✅ QA engine ready!")
    return _qa_chain


# ─── Main (for testing) ───────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "年假怎么休？"
    print(f"\n❓ Question: {query}\n")
    chain = get_qa_chain()
    answer = chain.invoke(query)
    print(f"💡 Answer:\n{answer}")
