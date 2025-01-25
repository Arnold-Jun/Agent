import os
import re
import chromadb
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from typing import Any
from setup import setup
from llama_index.core import SimpleDirectoryReader
from config import config
import pickle
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes

def load_and_index(
    update: bool,
    read_only: bool,
    data_dir: str,
    pipeline_cache_path: str,
    text_spliter: str = "sentence_splitter",
    text_spliter_args: dict[str, Any] = {},
    extractors: list[str] = [],
    pipeline_workers: int = 1,
):
    trans = []

    supported_extractors = ["title", "keyword", "questions_answered", "summary", "semantic"]
    for e in extractors:
        if e not in supported_extractors:
            raise ValueError(f"Unsupported extractor: {e}")

    # extracts a title over the context of each Node
    if "title" in extractors:
        from llama_index.core.extractors import TitleExtractor
        trans.append(TitleExtractor())


    if text_spliter == "sentence_splitter":
        from llama_index.core.node_parser import SentenceSplitter
        trans.append(SentenceSplitter(**text_spliter_args))
    elif text_spliter == "semantic":
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        trans.append(SemanticSplitterNodeParser(**text_spliter_args))
    else:
        raise ValueError(f"Unsupported text_splitter: {text_spliter}")


    if "keyword" in extractors:
        from llama_index.core.extractors import KeywordExtractor
        trans.append(KeywordExtractor())

    # extracts a set of questions that each Node can answer
    if "questions_answered" in extractors:
        from llama_index.core.extractors import QuestionsAnsweredExtractor
        trans.append(QuestionsAnsweredExtractor())

    # automatically extracts a summary over a set of Nodes
    if "summary" in extractors:
        from llama_index.core.extractors import SummaryExtractor
        trans.append(SummaryExtractor())

    trans.append(TextCleaner())
    trans.append(Settings.embed_model)

    db = chromadb.PersistentClient(
        path=config.chroma_db, settings=chromadb.Settings(allow_reset=True)
    )
    db.reset()  # Clear previously stored data in vector database
    chroma_collection = db.get_or_create_collection("dku_html_pdf")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=trans,
        vector_store=vector_store,
    )
    if os.path.exists(pipeline_cache_path):
        pipeline.load(pipeline_cache_path)

    documents_path = config.documents_path
    with open(documents_path, 'rb') as f:
        documents = pickle.load(f)
    nodes = pipeline.run(
        documents=documents, num_workers=pipeline_workers, show_progress=True
    )
    pipeline.persist(pipeline_cache_path)
    print("nodes over")

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(config.docstore_path)
    print("docstore over")


def main():
    setup(add_system_prompt=True)

    load_and_index(
        update=False,
        read_only=False,
        data_dir=str(config.data_dir),
        pipeline_cache_path=str(config.pipeline_cache),
        text_spliter="sentence_splitter",
        text_spliter_args={"chunk_size": 1024, "chunk_overlap": 20},
        extractors=[],
        pipeline_workers=1,
    )


if __name__ == "__main__":
    main()
