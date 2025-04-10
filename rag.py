# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import logging
import os

os.environ["OPENAI_API_KEY"] = ""

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "mistral:latest", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        # self.model = ChatOllama(model=llm_model)
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less. If you are not confident in any answer based on the provided context, say "I'm not sure about that" exactly, and then provide your best guess at the.
            """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        docs = filter_complex_metadata(docs)

        self.vector_store = Chroma(collection_name="full_documents",
            embedding_function=self.embeddings,
            persist_directory="chroma_db",
        )
        self.store = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.20},
                vectorstore=self.vector_store,
                docstore=self.store,
                child_splitter=self.text_splitter,
            )
        self.retriever.add_documents(docs)
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ingest_anytype(self, jsonDocs):
        docs = []

        for page in jsonDocs:
            page_id = page.get("id", "")
            title = page.get("name", "")
            snippet = page.get("snippet", "")

            # Extract visible text from blocks
            blocks = page.get("blocks", [])
            block_texts = [
                block.get("text", {}).get("text", "")
                for block in blocks
                if "text" in block and block["text"].get("text")
            ]
            full_content = "\n".join([snippet] + block_texts).strip()

            # Extract metadata
            tags = [
                tag["name"] for tag in page.get("details", []) 
                if tag["id"] == "tags"
                for tag in tag.get("details", {}).get("tags", [])
            ]

            author = next((
                detail["details"]["details"].get("name", "Unknown")
                for detail in page.get("details", [])
                if detail["id"] == "created_by"
            ), "Unknown")

            created = next((
                detail["details"].get("created_date")
                for detail in page.get("details", [])
                if detail["id"] == "created_date"
            ), None)

            docs.append(Document(
                page_content=full_content,
                metadata={
                    "id": page_id,
                    "title": title,
                    "author": author,
                    "created_date": created,
                    "tags": tags,
                    "space_id": page.get("space_id")
                }
            ))
        docs = filter_complex_metadata(docs)

        self.vector_store = Chroma(collection_name="full_documents",
            embedding_function=self.embeddings,
            persist_directory="chroma_db",
        )
        self.store = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.20},
                vectorstore=self.vector_store,
                docstore=self.store,
                child_splitter=self.text_splitter,
            )
        self.retriever.add_documents(docs)
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
             self.retriever = ParentDocumentRetriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
                vectorstore=self.vector_store,
                docstore=self.store,
                child_splitter=self.text_splitter,
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."
        
        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        answer = chain.invoke(formatted_input)

        return answer, retrieved_docs

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
