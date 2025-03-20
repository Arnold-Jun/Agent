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


class KeywordRetriever(dspy.Module):
    """Retrieve texts from the database that contain the same keywords in the query which related to Duke Kunshan University."""

    def __init__(self, retriever_top_k: int = 10, reranker_top_n: int = 3):
        super().__init__()
        self.client = Redis.from_url("redis://default:WnJU4r2ROwQUB2qztvAJ3wCQbrCNksRr@redis-10193.c44.us-east-1-2.ec2.redns.redis-cloud.com:10193")
        self.retriever_top_k = retriever_top_k

        schema = IndexSchema.from_yaml(
            os.path.join(config.module_root_dir, "custom_schema.yaml")
        )
        self.index_name = schema.index.name

    def forward(
        self,
        query: Annotated[
            str,
            Field(
                description="Keywords that might appear in the answer to the question."
            ),
        ],
        internal_memory: dict,
    ):
        # Escape all punctuations, e.g. "can't" -> "can\'t"
        def escape_strs(strs: list[str]):
            pattern = f"[{re.escape(string.punctuation)}]"
            return [
                re.sub(pattern, lambda match: f"\\{match.group(0)}", s) for s in strs
            ]

        # Filters stop words
        def filter_stopwords(tokens: list[str]):
            return [word for word in tokens if word.lower() not in self.stopwords]

        exclude = list(internal_memory.get("ids", set()))

        try:
            nltk.data.find("tokenizers/punkt_tab/english")
        except LookupError:
            nltk.download("punkt_tab")
        # Break down the query into tokens
        tokens = word_tokenize(query)
        # Remove tokens that are PURELY punctuations
        orig_keywords = list(
            filter(lambda token: token not in string.punctuation, tokens)
        )
        orig_keywords = filter_stopwords(orig_keywords)
        orig_keywords = escape_strs(orig_keywords)

        keywords = []
        weights = []
        TUPLE_LIMIT = 4
        BOOST_FACTOR = 2
        for i in range(1, TUPLE_LIMIT + 1):
            for combo in combinations(orig_keywords, i):
                keywords.append(" ".join(combo))
                weights.append(BOOST_FACTOR ** (i - 1))

        # `|` means searching the union of the words/tokens.
        # `%` means fuzzy search with Levenshtein distance of 1.
        # Query attributes are used here to set the weight of the keywords.
        text_str = " | ".join(
            [
                f"({keyword}) => {{ $weight: {weight} }}"
                for keyword, weight in zip(keywords, weights)
            ]
        )
        query_str = "@text:(" + text_str + ")"

        exclude = escape_strs(exclude)
        exclude_str = " ".join([f"-@id:({e})" for e in exclude])
        if exclude_str:
            query_str += " " + exclude_str

        retriever_top_k = 5
        query_cmd = (
            Query(query_str).scorer("BM25").paging(0, retriever_top_k).with_scores()
        )

        results = self.client.ft(self.index_name).search(query_cmd)
        try:
            nodes = [
                NodeWithScore(
                    node=TextNode(
                        id=r.id, text=r.text, metadata={"file_path": r.file_path}
                    ),
                    score=r.score,
                )
                for r in results.docs
            ]
        except:
            nodes = [
                NodeWithScore(node=TextNode(id=r.id, text=r.text), score=r.score)
                for r in results.docs
            ]

        nodes = simplify_nodes(nodes)
        result = nodes_to_dicts(nodes)

        return dspy.Prediction(
            result=result, internal_result={"ids": {r.id for r in results.docs}}
        )

