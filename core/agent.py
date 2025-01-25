from typing import Any
import traceback
from llama_index.core import Settings
from llama_index.core.base.llms.types import CompletionResponse
import functools
from dsp import LM
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import core.dspy_patch
from core.tools.custom_llama_index import VectorRetriever
from core.dspy_classes.plan import Planner
from core.dspy_classes.conversation_memory import ConversationMemory
from core.dspy_classes.tool_memory import ToolMemory
from core.dspy_classes.query_rewrite import QueryRewrite
from core.dspy_classes.prompt_settings import VERBOSE
from core.dspy_classes.synthesizer import Synthesizer
from core.dspy_classes.judge import Judge
from config import config
from setup import setup


class CustomClient(LM):
    def __init__(self) -> None:
        self.provider = "default"
        self.history = []
        self.kwargs = {
            "max_tokens": config.context_window,
        }

    def basic_request(self, prompt: str, **kwargs: Any) -> CompletionResponse:

        response = Settings.llm.complete(prompt, **kwargs)
        return response

    def inspect_history(self, n: int = 1, skip: int = 0) -> str:
        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]
            if prompt != last_prompt:
                printed.append((prompt, x["response"].text))
            last_prompt = prompt
            if len(printed) >= n:
                break

        printing_value = ""
        for idx, (prompt, text) in enumerate(reversed(printed)):
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n"
            printing_value += prompt
            printing_value += self.print_green(text, end="")
            printing_value += "\n\n\n"

        print(printing_value)
        return printing_value

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        return [self.request(prompt, **kwargs).text]


class Agent(dspy.Module):
    def __init__(
        self,
        max_iterations: int = 5,
        streaming: bool = False,
        get_intermediate: bool = False,
        rewrite_query: bool = False,
    ):
        """
        Args:
            max_iterations: The maximum rounds of tool call/evaluation the agent
                could execute for a user message. This includes the first round
                of tool calls with the initial user message.
            streaming: If `True`, returns the LLM response as a streaming generator
                for `reponse` returned by synthesizer, else simply return the
                complete response as a string.
            get_itermediate: If `True`, `forward()` would return the synthesized
                result for each agent iteration as a generator.
        """

        super().__init__()
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.get_intermediate = get_intermediate
        self.rewrite_query = rewrite_query

        self.planner = assert_transform_module(
            Planner([VectorRetriever()]),
            functools.partial(backtrack_handler, max_backtracks=5),
        )
        self.conversation_memory = ConversationMemory()
        self.tool_memory = ToolMemory()

        self.internal_memory = {}
        self.synthesizer = Synthesizer()
        self.judge = assert_transform_module(
            Judge(), functools.partial(backtrack_handler, max_backtracks=5)
        )
        self.queryrewriter = QueryRewrite()

        self.prev_response = None

    def reset(self):
        self.prev_response = None
        self.conversation_memory = ConversationMemory()

    def _forward_gen(self, current_user_message: str, question_id: str):

        limits = self.planner.get_token_limits()

        # Reset tool memory for each user message
        self.tool_memory.reset()

        # Clear internal memory for each user message
        self.internal_memory.clear()

        # Add previous response to conversation memory
        if self.prev_response is not None:
            if self.streaming:
                r = self.prev_response.get_full_response()
            else:
                r = self.prev_response
            self.conversation_memory(
                role="assistant",
                content=r,
                max_history_size=limits["conversation_history"],
            )
        # Deal with DSPy assertions
        # Reference: https://github.com/stanfordnlp/dspy/blob/af5186cf07ab0b95d5a12690d5f7f90f202bc86e/dspy/predict/retry.py#L59
        with dspy.settings.lock:
            dspy.settings.backtrack_to = None

        for (name, model), tool in zip(
            self.planner.name_to_model.items(), self.planner.tools
        ):

            r = tool(
                query=current_user_message, internal_memory=self.internal_memory
            )
            first_itr_result, internal_result = r.result, r.internal_result
            if "ids" in internal_result:
                self.internal_memory["ids"] = (
                    self.internal_memory.get("ids", set()) | internal_result["ids"]
                )

            self.tool_memory(
                current_user_message=current_user_message,
                conversation_memory=self.conversation_memory,
                calls=[model(name=name, params={"query": current_user_message})],
                result=first_itr_result,
                max_history_size=limits["tool_history"],
            )


        synthesizer_args = dict(
            current_user_message=current_user_message,
            conversation_memory=self.conversation_memory,
            tool_memory=self.tool_memory,
            streaming=self.streaming,
        )

        # The subsequent rounds of tool calling
        for i in range(self.max_iterations - 1):
            if self.get_intermediate:
                result = self.synthesizer(**synthesizer_args)
                yield result

            judgement = self.judge(
                current_user_message=current_user_message,
                conversation_memory=self.conversation_memory,
                tool_memory=self.tool_memory,
            ).judgement
            if judgement:
                break

            if self.rewrite_query:
                query = self.queryrewriter(
                    current_user_message=current_user_message,
                    conversation_memory=self.conversation_memory,
                    tool_memory=self.tool_memory,
                ).rewritten_query
            else:
                query = current_user_message

            try:
                p = self.planner(
                    current_user_message=query,
                    conversation_memory=self.conversation_memory,
                    tool_memory=self.tool_memory,
                    max_calls=self.max_iterations - i,
                )

            except dspy.DSPyAssertionError:
                if VERBOSE:
                    print("max assertion retries hit")
                break

            r = p.tool(
                **p.calls[0].params.model_dump(),
                internal_memory=self.internal_memory,
            )

            resuzlt, internal_result = r.result, r.internal_result
            if "ids" in internal_result:
                self.internal_memory["ids"] = (
                    self.internal_memory.get("ids", set()) | internal_result["ids"]
                )

            self.tool_memory(
                current_user_message=current_user_message,
                conversation_memory=self.conversation_memory,
                calls=p.calls,
                result=result,
                max_history_size=limits["tool_history"],
            )


        self.prev_response = self.synthesizer(
            **synthesizer_args
        ).response

        self.conversation_memory(
            role="user",
            content=current_user_message,
            max_history_size=limits["conversation_history"],
        )
        yield dspy.Prediction(response=self.prev_response)

    def forward(self, current_user_message: str, question_id: str = ""):
        gen = self._forward_gen(current_user_message, question_id)
        if self.get_intermediate:
            return gen
        else:
            for i in gen:
                return i


def main():
    setup()

    llama_client = CustomClient()
    dspy.settings.configure(lm=llama_client)
    import time

    agent = Agent(max_iterations=2, streaming=True, get_intermediate=False, rewrite_query=False)

    while True:
        try:
            print("*" * 10)
            current_user_message = input("Enter your query about DKU: ")
            start_time = time.time()
            responses_gen = agent(current_user_message=current_user_message)
            first_token = True
            print("Response:")
            for r in responses_gen.response:
                if first_token:
                    end_time = time.time()
                    print(f"first token时间:{end_time-start_time}")
                    first_token = False
                print(r, end="")
            print()


        except EOFError:
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())

    input()
