import glob
import os
from haystack import Pipeline, Document
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from haystack.components.preprocessors import DocumentSplitter


if not os.path.exists("data"):
    os.makedirs("data")

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
)
# create an indexing pipeline
file_converter = TextFileToDocument()
# 250 words 30 overlaping
splitter = DocumentSplitter(split_by="word", split_length=250, split_overlap=30) 
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)


indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", file_converter)
indexing_pipeline.add_component("splitter", splitter) # Adicionado
indexing_pipeline.add_component("embedder", doc_embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

# 3. Add documents to the pipeline
#documents_to_index = [
#    Document(content="Títulos Públicos	Empréstimos ao governo em troca de juros.	Considerados de baixo risco.	Retorno menor em comparação com ações."),
#    Document(content="Fundos Imobiliários	Investimentos em propriedades por meio de cotas.	Renda passiva e diversificação.	Taxas de administração e exposição ao mercado imobiliário."),
#    Document(content="Ações	Participações em empresas negociadas na bolsa.	Alto potencial de retorno.	Risco elevado e volatilidade do mercado."),
#]

file_paths = [os.path.join("data", f) for f in os.listdir("data")]


indexing_pipeline.run({"converter": {"sources": file_paths}})

print(f"Successfully indexed {document_store.count_documents()} documents.")
