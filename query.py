# import langchain library

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.llm import LLMChain
from langchain.text_splitter import MarkdownTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate

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
from chat_vector_db import MyConversationalRetrievalChain
from common.utils import OPENAI_API_KEY
from stuff import CustomStuffDocumentsChain

# path saved prompt template
prompt_path = 'prompts'

template_prompt_path = os.path.join(prompt_path, 'template_prompt.txt')
system_prompt_path = os.path.join(prompt_path, 'system_prompt.txt')
human_prompt_path = os.path.join(prompt_path, 'human_prompt.txt')

# Read the content of each file
with open(template_prompt_path, 'r') as file:
    _template = file.read()

CONDENSE_QUESTION_ENHANCED_PROMPT = PromptTemplate.from_template(_template)

with open(system_prompt_path, 'r') as file:
    system_template = file.read()

with open(human_prompt_path, 'r') as file:
    human_template = file.read()

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
]

DOC_CHAIN_PROMPT = ChatPromptTemplate.from_messages(messages)


# create a ChatVectorDBChain for question/answer sessions
def get_chain(
        vectorstore: VectorStore, question_handler, stream_handler,
        tracing: bool = False):
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation

    # Set up managers for callbacks
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])

    # Maximum limit of document sources
    max_source_document_limit = 3

    # If tracing enabled, set up tracing for debugging purposes
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    # set up  model for question generation
    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key=OPENAI_API_KEY,
    )

    # set up streaming for doc combine
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    # create llm chain for doc combine
    llm_doc_chain = LLMChain(
        llm=streaming_llm, prompt=DOC_CHAIN_PROMPT, verbose=False,
        callback_manager=manager
    )

    # create doc chain for assembling doc details
    doc_chain = CustomStuffDocumentsChain(
        llm_chain=llm_doc_chain,
        document_variable_name="context",
        verbose=False,
        callback_manager=manager
    )

    # set up question generator chain
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_ENHANCED_PROMPT,
        callback_manager=manager
    )

    # create a customized QA retrieval chain
    qa_chain = MyConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True,
        max_tokens_limit=max_source_document_limit
    )
    return qa_chain


def get_vector_store(persist_directory: str,
                     documents_path: str, reindex=False) -> VectorStore:
    # embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #  evaluate reindex
    if not reindex and os.path.exists(persist_directory):
        return Chroma(embedding_function=embeddings,
                      persist_directory=persist_directory)

    # load contents
    loader = DirectoryLoader(documents_path, loader_cls=TextLoader)

    # split into chunks
    documents = loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=1500,
                                         chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings,
                                 persist_directory=persist_directory)
