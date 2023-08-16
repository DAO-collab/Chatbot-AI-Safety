from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """You are Lexibot, an helpful assistant specializing in legal cases. You are a world expert in law. 
    You analyze legal cases and reply to the user in a helpful way. You can use the following pieces of context to answer 
    the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    {chat_history}

    You are Lexibot, an helpful assistant specializing in legal cases. You are a world expert in law. You analyze legal 
    cases and answer user questions in a helpful way. You can use the above pieces of context to reply to the user. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Reply using the context above to help you but it might not be relevant.
    Reply as the world-class legal expert, Lexibot:"""
messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_chain(
        vectorstore: VectorStore, question_handler, stream_handler,
        tracing: bool = False):
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key=OPENAI_API_KEY,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=prompt,
        callback_manager=manager
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True
    )
    return qa


def get_vector_store(persist_directory: str,
                     documents_path: str, reindex=False) -> VectorStore:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if not reindex and os.path.exists(persist_directory):
        return Chroma(embedding_function=embeddings,
                      persist_directory=persist_directory)

    loader = DirectoryLoader(documents_path, loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=1500,
                                         chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings,
                                 persist_directory=persist_directory)
