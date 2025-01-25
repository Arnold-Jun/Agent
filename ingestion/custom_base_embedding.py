from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import (
    Field,
    ConfigDict,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.schema import TransformComponent
from llama_index.core.utils import get_tqdm_iterable
import llama_index.core.instrumentation as instrument
from concurrent.futures import ThreadPoolExecutor, as_completed
dispatcher = instrument.get_dispatcher(__name__)
Embedding = List[float]
class BaseEmbedding(TransformComponent, DispatcherSpanMixin):
    """Base class for embeddings."""

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )

    @dispatcher.span
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = True,) -> List[Embedding]:
        """Get a list of text embeddings, each text processed individually, with progress tracking."""

        queue_with_progress = enumerate(get_tqdm_iterable(texts, show_progress, "Generating embeddings"))

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        def process_text(text: str) -> Embedding:
            embeddings = self._get_text_embeddings([text])
            return embeddings[0]

        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(lambda x: process_text(x[1]), queue_with_progress))

        return embeddings

    # @dispatcher.span
    # def get_text_embedding_batch(
    #         self,
    #         texts: List[str],
    #         show_progress: bool = False,
    #         **kwargs: Any,
    # ) -> List[Embedding]:
    #     """Get a list of text embeddings, with batching."""
    #     cur_batch: List[str] = []
    #     result_embeddings: List[Embedding] = []
    #
    #     queue_with_progress = enumerate(
    #         get_tqdm_iterable(texts, show_progress, "Generating embeddings")
    #     )
    #
    #     model_dict = self.to_dict()
    #     model_dict.pop("api_key", None)
    #
    #     def process_batch(batch: List[str]) -> List[Embedding]:
    #         embeddings = self._get_text_embeddings(batch)
    #         return embeddings
    #
    #     with ThreadPoolExecutor() as executor:
    #         for idx, text in queue_with_progress:
    #             cur_batch.append(text)
    #             if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
    #                 # flush
    #                 dispatcher.event(
    #                     EmbeddingStartEvent(
    #                         model_dict=model_dict,
    #                     )
    #                 )
    #                 with self.callback_manager.event(
    #                         CBEventType.EMBEDDING,
    #                         payload={EventPayload.SERIALIZED: self.to_dict()},
    #                 ) as event:
    #                     embeddings = list(executor.map(process_batch, [cur_batch]))
    #                     result_embeddings.extend(embeddings[0])
    #                     event.on_end(
    #                         payload={
    #                             EventPayload.CHUNKS: cur_batch,
    #                             EventPayload.EMBEDDINGS: embeddings[0],
    #                         },
    #                     )
    #                 dispatcher.event(
    #                     EmbeddingEndEvent(
    #                         chunks=cur_batch,
    #                         embeddings=embeddings[0],
    #                     )
    #                 )
    #                 cur_batch = []
    #     print(len(result_embeddings))
    #     return result_embeddings
