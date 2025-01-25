from itertools import takewhile
from llama_index.core import Settings
import dspy
from core.utils import token_limit_ratio_to_count, truncate_tokens_all
from core.dspy_common import custom_cot_rationale, get_template
from core.dspy_classes.prompt_settings import (
    CURRENT_USER_MESSAGE_FIELD,
    CONVERSATION_HISTORY_FIELD,
    CONVERSATION_SUMMARY_FIELD,
    TOOL_HISTORY_FIELD,
    TOOL_SUMMARY_FIELD,
    ROLE_PROMPT,
)
from core.dspy_classes.conversation_memory import ConversationMemory
from core.dspy_classes.plan import ToolMemory

from config import config

from datetime import date


def make_synthesizer_signature():

    fields = {
        "current_user_message": (str, CURRENT_USER_MESSAGE_FIELD),
        "conversation_history": (str, CONVERSATION_HISTORY_FIELD),
        "conversation_summary": (str, CONVERSATION_SUMMARY_FIELD),
        "tool_history": (str, TOOL_HISTORY_FIELD),
        "tool_summary": (str, TOOL_SUMMARY_FIELD),
        "response": (
            str,
            dspy.OutputField(desc="You response to the Current User Message."),
        ),
    }
    current_date = date.today()

    # instruction = "Your current task is to answer the Current User Message according to your Tool Memory."
    instruction = (
        "Your current task is to answer the Current User Message according to your Tool Memory."
        "Your answer should be as detailed as possible, taking advantage of the relevant context in tool memory."
        "Your answer should be be organized and use bullet points if needed."
        "The contexts might contain unrelated information or non-DKU resources. "
        "Always prefer DKU resources first. "
        "You may include other resources (including even Duke resources) only as "
        "a second option unless directly asked, or that resource is clearly "
        "available to the DKU community via means such as a partnership with DKU. "
        "The source of contexts is contained in the url in metadata,"
        "Include the urls to the sources used in your answer at the end, like 'reference links:'. "
        "Do not include the urls to the sources that you did not use in your answer. "
        "The link needs to be markdown so that it can be clicked, and the text shown is a "
        "summary of the link, make sure the text is accurate about the url, and please don't print duplicate links. "
        "make sure the reference link you offer is the accurate copy from your database. "
        "If you see 'no url' for a source, do not provide the link. "
        "Do not use the url of one source for another source, and do not guess the url. "
        "Your internal operation should also not be transparent to the user, "
        '"do not include phrases like "Based on the conversation history", '
        '"Based on the information retrieved from the Tool History and Conversation History", "According to the tool history" in your answer. '
        "When you're asked a general question, automatically change it to something DKU related, "
        "like 'what does CTL do?' to 'what does CTL do at DKU?' "
        "If the Current User Message is ambiguous, you may first try to answer it to the best extent "
        "with the known information, then ask the user for further clarifications. "
        "Additionally, you should point out the cases where the information in Tool Memory does not "
        "adequately address the Current User Message. "
        "Please remember do not provide the same information repeatedly!"
        ### time ...
        f"Today's date is {current_date}. For timeliness issues, please consider more relevant context closer to the current date."
    )

    return dspy.make_signature(
        fields, ROLE_PROMPT + "\n\n" + instruction, "SynthesizerSignature"
    )


SynthesizerSignature = make_synthesizer_signature()


class ResponseGen:
    """A generator that extracts `response` field and strips whitespace
    given the generator for the entire LLM completion.
    """

    def __init__(
        self, prompt: str
    ):
        self.llm_completion_gen = Settings.llm.stream_complete(prompt)
        self.full_response = ""

    def __iter__(self):

        def rstripped(s):
            """Extract the trailing whitespace itself."""
            return "".join(reversed(tuple(takewhile(str.isspace, reversed(s)))))

        field = "Response:"
        before_response = ""
        for r in self.llm_completion_gen:
            before_response += r.delta
            offset = before_response.find(field)
            if offset != -1:
                s = before_response[offset + len(field) :]
                if s.strip():
                    self.full_response += s.strip()
                    yield s.strip()
                    prev_whitespace = rstripped(s)
                    break


        for r in self.llm_completion_gen:
            s = r.delta
            self.full_response += prev_whitespace + s.rstrip()

            yield prev_whitespace + s.rstrip()

            prev_whitespace = rstripped(s)



    def get_full_response(self) -> str:
        # Make sure the entire response is read
        for _ in self:
            pass
        return self.full_response


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(
            SynthesizerSignature, rationale_type=custom_cot_rationale
        )
        self.token_ratios: dict[str, float] = {
            "current_user_message": 2 / 15,
            "conversation_history": 2 / 15,
            "conversation_summary": 1 / 15,
            "tool_history": 5 / 15,
            "tool_summary": 1 / 15,
        }

    def get_token_limits(self) -> dict[str, int]:
        return token_limit_ratio_to_count(
            self.token_ratios, len(get_template(self.synthesizer))
        )

    def forward(
        self,
        current_user_message: str,
        conversation_memory: ConversationMemory,
        tool_memory: ToolMemory,
        streaming: bool,
    ):


        synthesizer_args = dict(
            current_user_message=current_user_message,
            conversation_history=conversation_memory.history_str(),
            conversation_summary=conversation_memory.summary,
            tool_history=tool_memory.history_str(),
            tool_summary=tool_memory.summary,
        )
        synthesizer_args = truncate_tokens_all(
            synthesizer_args, self.get_token_limits()
        )

        if streaming:
            synthesizer_template = get_template(
                self.synthesizer, **synthesizer_args
            )
            response_gen = ResponseGen(synthesizer_template)
            return dspy.Prediction(response=response_gen)

        else:

            response = self.synthesizer(**synthesizer_args).response
            return dspy.Prediction(response=response)
