from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter

from haystack.components.rankers import TransformersSimilarityRanker

from itertools import chain
from typing import Any

from haystack import component
from haystack.core.component.types import Variadic

from typing import List

@component
class ListJoiner:
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}


memory_store = InMemoryChatMessageStore()
memory_retriever = ChatMessageRetriever(memory_store)
memory_writer = ChatMessageWriter(memory_store)


embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
)

template = [
    ChatMessage.from_system(
        """
        You are a virtual financial EDUCATOR, not a financial advisor, planner, or assistant. Your sole purpose is to explain financial concepts objectively. You do not help users manage their money or make financial decisions.
        Do not provide personalized investment advice or asset recommendations. Never evaluate, reference, or incorporate the user's specific age, income, risk tolerance, capital, or current portfolio into your answer.
        Never use the word "you" in the context of taking action. Speak in generalizations. (Use "Investors often consider..." instead of "You should consider...").
        Do not tell the user which asset, strategy, or account is "better" for them. Explain the objective mechanics, pros, and cons of the concepts, and let the user decide.
        Answer the user's question using primarely the content provided in the <documents> section.
        You may use your internal knowledge to understand acronyms or translate the user's question to match the documents.
        If the answer is not in the documents, inform the user that you don't have the information.
        Answer in the same language as the User's Question.
        Keep answers brief and direct.
        """
    ),
    ChatMessage.from_user(
        """
        <documents>
        {% for document in documents %}
            <doc>
            {{ document.content }}
            </doc>
        {% endfor %}
        </documents>

        <conversation_history>
        {% for memory in memories[-3:] %}
            {{ memory.text }}
        {% endfor %}
        </conversation_history>

        Question: {{question}}
        CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
        BANNED WORDS: DO NOT use the words "you", "your", or "yours" when discussing financial actions. Speak entirely in the abstract third-person (e.g., use "Investors often...", "An individual might...", or "A portfolio...").
        BANNED CONTEXT: DO NOT reference the user's specific age, income, dollar amounts, current portfolio, or personal situation in your answer. Abstract the math or concepts generally.
        BANNED DIRECTIVES: DO NOT use advisory words like "consider", "should", "buy", "sell", "start by", or "recommend". NEVER tell the user what to do with their money or suggest a specific action.
        NO CONDITIONAL ADVICE: DO NOT state which asset, account, or strategy is "better" or "more suitable" for specific types of risk tolerances. Just state the objective mechanics, pros, and cons.

        """
    )
]

# Create document store and save documents with embeddings
text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
# TOP K 20
retriever = MilvusEmbeddingRetriever(document_store=document_store, top_k=20)

#re-ranker
#select the best 5 from 20 documents

ranker = TransformersSimilarityRanker(
    model="BAAI/bge-reranker-base",
    top_k=5,
    meta_fields_to_embed=[] # ignore metadata during ranking
)

prompt_builder = ChatPromptBuilder(template=template, required_variables=["question"])
chat_generator = OllamaChatGenerator(
    model="qwen3:14b", url="http://127.0.0.1:11434",
    timeout=600,
    generation_kwargs={
        "temperature": 0.0,  #STRICT AND DETERMINISTIC, no creativity
        "top_p": 0.9         #NEEDS FURTHER RESEARCH
    }
)

# 4. Build the RAG pipeline
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("ranker", ranker)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# components for memory
basic_rag_pipeline.add_component("memory_retriever", memory_retriever)
basic_rag_pipeline.add_component("memory_writer", memory_writer)
basic_rag_pipeline.add_component("memory_joiner", ListJoiner(List[ChatMessage]))

# 5. Connect the components
basic_rag_pipeline.connect("memory_joiner", "memory_writer")
basic_rag_pipeline.connect("memory_retriever", "prompt_builder.memories")
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
#reranker things
basic_rag_pipeline.connect("retriever.documents", "ranker.documents")
basic_rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

def run_pipeline(question, session_id="default_session"):
    response = basic_rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
            #ranker needs this to work
            "ranker": {"query": question},
            "memory_joiner": {"values": [ChatMessage.from_user(question)]},
            "memory_retriever": {"chat_history_id": session_id},
            "memory_writer": {"chat_history_id": session_id}
        }
    )

    return response["llm"]["replies"][0].text

def run_pipeline_for_evaluation(question):
    response = basic_rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "ranker": {"query": question},
            "memory_joiner": {"values":[ChatMessage.from_user(question)]}
        },
        include_outputs_from={"ranker"}
    )

    # LLM answer
    answer = response["llm"]["replies"][0].text

    # contexts from the ranker
    documents = response["ranker"]["documents"]
    contexts =[doc.content for doc in documents]

    return answer, contexts
