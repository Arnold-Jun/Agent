from typing import Annotated
from enum import Enum
from collections.abc import Iterator
from pydantic import Field
import dspy
from core.utils import truncate_tokens
from core.dspy_common import custom_cot_rationale
import chromadb
import llama_index
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.legacy import VectorStoreIndex
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.legacy.schema import TextNode, NodeWithScore, MetadataMode
from llama_index.legacy.node_parser.text.token import TokenTextSplitter
from llama_index.legacy.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from typing import Mapping, Any
from config import config
import pandas as pd
import re



def get_reranker(top_n: int):
    return ColbertRerank(
        top_n=top_n,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )

def get_url(metadata):
    try:
        try:
            path = metadata["file_path"]
        except:
            path = metadata["file_directory"] + "/" + metadata["filename"]
        if "dku_website" in path:
            match = re.search(r"dku_website/.*", path)
            if match:
                result = match.group(0)
                matching_row = df[df["file_path"] == result]
                if not matching_row.empty:
                    return matching_row.iloc[0]["url"]
        elif "new_bulletin" in path:
            match = re.search(r"new_bulletin/.*", path)
            if match:
                result = match.group(0)
                matching_row = df[df["file_path"] == result]
                if not matching_row.empty:
                    return matching_row.iloc[0]["url"]
        return "no url"
    except Exception as e:
        return f"no url, error: {str(e)}"


def simplify_nodes(nodes: list[NodeWithScore]) -> NodeWithScore:
    return [
        NodeWithScore(
            node=TextNode(
                node_id=node.node_id,
                text=node.text,
                metadata={"url": get_url(node.metadata)},
            ),
            score=node.score,
        )
        for node in nodes
    ]


def nodes_to_dicts(nodes: list[NodeWithScore]):
    return [{"text": node.text, "metadata": node.metadata} for node in nodes]



class VectorRetriever(dspy.Module):
    """Retrieve texts from the database that are semantically similar to the query."""

    def __init__(
        self,
        retriever_top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_n: int = 5,
    ):
        self.retriever_top_k = retriever_top_k
        self.use_reranker = use_reranker
        self.reranker_top_n = reranker_top_n

        db = chromadb.PersistentClient(path=config.chroma_db)
        chroma_collection = db.get_collection("dku_html_pdf")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    def forward(
        self,
        query: Annotated[
            str,
            Field(
                description="Texts that might be semantically similar to the real answer to the question."
            ),
        ],
        internal_memory: dict,
    ):
            exclude = list(internal_memory.get("ids", set()))
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="id", value=i, operator=FilterOperator.NE)
                    for i in exclude
                ]
            )
            retriever = self.index.as_retriever(
                similarity_top_k=self.retriever_top_k, filters=filters
            )

            retrieved_nodes = retriever.retrieve(
                truncate_tokens(query, 7000)
            )

            if self.use_reranker:
                reranker = get_reranker(self.reranker_top_n)
                nodes = reranker.postprocess_nodes(
                    retrieved_nodes,
                    query_str=truncate_tokens(
                        query, 500, tokenizer=reranker._tokenizer
                    ),
                )
            else:
                nodes = retrieved_nodes

            nodes = simplify_nodes(nodes)
            result = nodes_to_dicts(nodes)

            return dspy.Prediction(
                result=result,
                internal_result={"ids": {r.node_id for r in nodes}},
            )


