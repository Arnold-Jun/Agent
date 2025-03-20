from typing import Any
import traceback
from llama_index.core import Settings
from llama_index.core.base.llms.types import CompletionResponse
import functools
from dsp import LM
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import core.dspy_patch
from core.tools.custom_llama_index import VectorRetriever, KeywordRetriever
from core.dspy_classes.plan import Planner
from core.dspy_classes.conversation_memory import ConversationMemory
from core.dspy_classes.tool_memory import ToolMemory
from core.dspy_classes.query_rewrite import QueryRewrite
from core.dspy_classes.prompt_settings import VERBOSE
from core.dspy_classes.synthesizer import Synthesizer
from core.dspy_classes.judge import Judge
from tools.email.email import EmailTools
from tools.search.python_googlesearch import GoogleSearch
from config import config
from setup import setup
from dspy_classes.polish import Polish
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
            polish: bool = False
    ):
        """
        初始化Agent类，配置最大迭代次数、是否使用流式输出、是否获取中间结果等参数。
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.get_intermediate = get_intermediate
        self.rewrite_query = rewrite_query
        self.polish = polish

        self.planner = assert_transform_module(
            Planner([VectorRetriever(), KeywordRetriever() EmailTools(), GoogleSearch()]),
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
        self.polisher = Polish()

        self.prev_response = None
        self.tool_cache = {}  # 新增缓存工具结果

    def reset(self):
        self.prev_response = None
        self.conversation_memory = ConversationMemory()
        self.tool_cache.clear()  # 重置缓存

    def get_tool_result(self, tool, tool_name, query, internal_memory):
        if tool_name in self.tool_cache:
            return self.tool_cache[tool_name]

        result = tool(query=query, internal_memory=internal_memory)
        self.tool_cache[tool_name] = result
        return result

    def _forward_gen(self, current_user_message: str, question_id: str):

        limits = self.planner.get_token_limits()

        self.tool_memory.reset()
        self.internal_memory.clear()

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

        # 初始化工具调用
        # for (name, model), tool in islice(zip(self.planner.name_to_model.items(), self.planner.tools), 1):
        #
        #     r = self.get_tool_result(tool, name, current_user_message, self.internal_memory)
        #     first_itr_result, internal_result = r.result, r.internal_result
        #     if "ids" in internal_result:
        #         self.internal_memory["ids"] = (
        #                 self.internal_memory.get("ids", set()) | internal_result["ids"]
        #         )
        #
        #     self.tool_memory(
        #         current_user_message=current_user_message,
        #         conversation_memory=self.conversation_memory,
        #         calls=[model(name=name, params={"query": current_user_message})],
        #         result=first_itr_result,
        #         max_history_size=limits["tool_history"],
        #     )

        synthesizer_args = dict(
            current_user_message=current_user_message,
            conversation_memory=self.conversation_memory,
            tool_memory=self.tool_memory,
            streaming=self.streaming,
        )

        # 后续迭代过程中的工具调用
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
            elif self.polish:
                query = current_user_message + " " + self.polisher(
                    current_user_message=current_user_message,
                    conversation_memory=self.conversation_memory,
                    tool_memory=self.tool_memory,
                ).output
                current_user_message += query
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

            result, internal_result = r.result, r.internal_result
            if "ids" in internal_result:
                self.internal_memory["ids"] = (
                        self.internal_memory.get("ids", set()) | internal_result["ids"]
                )

            if "search_url" in internal_result:
                self.internal_memory["search_url"] = list(set(self.internal_memory["search_url"] + internal_result["search_url"]))

            self.tool_memory(
                current_user_message=current_user_message,
                conversation_memory=self.conversation_memory,
                calls=p.calls,
                result=result,
                max_history_size=limits["tool_history"],
            )

        synthesizer_args = dict(
            current_user_message=current_user_message,
            conversation_memory=self.conversation_memory,
            tool_memory=self.tool_memory,
            streaming=self.streaming,
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

    agent = Agent(max_iterations=3, streaming=True, get_intermediate=False, rewrite_query=False, polish=True)

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
