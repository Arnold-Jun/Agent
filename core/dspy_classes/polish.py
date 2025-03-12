import dspy
from core.dspy_classes.prompt_settings import (
    CURRENT_USER_MESSAGE_FIELD,
    CONVERSATION_HISTORY_FIELD,
    CONVERSATION_SUMMARY_FIELD,
    TOOL_HISTORY_FIELD,
    TOOL_SUMMARY_FIELD,
)
from core.dspy_classes.conversation_memory import ConversationMemory
from core.dspy_classes.tool_memory import ToolMemory
from core.utils import truncate_tokens_all, token_limit_ratio_to_count
from core.dspy_common import get_template, custom_cot_rationale

local_lm = dspy.LM('openai/sglang/Llama-3.1-8B-Instruct', api_base="http://10.201.8.114:8003/v1", api_key="None", max_tokens=4000)
dspy.configure(lm=local_lm)

untrusted_lm = dspy.LM('openai/sglang/Llama-3.1-8B-Instruct', api_base="http://10.201.8.114:8003/v1", api_key="None", max_tokens=4000)

def make_prompt_creater_signature():
    """
    You are a helpful assistant that is very mindful of user privacy. You have access to a powerful large language model that you can query. Given a user request, create a prompt for your large language model that preserves user privacy, so that this model can help you complete the user request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY GENERATE A PROMPT.
    """
    fields = {
        "current_user_message": (str, CURRENT_USER_MESSAGE_FIELD),
        "conversation_history": (str, CONVERSATION_HISTORY_FIELD),
        "conversation_summary": (str, CONVERSATION_SUMMARY_FIELD),
        "tool_history": (str, TOOL_HISTORY_FIELD),
        "tool_summary": (str, TOOL_SUMMARY_FIELD),
        "created_prompt": (
            str,
            dspy.OutputField(),
        )
    }

    instruction = (
        "You are a privacy-conscious assistant utilizing an advanced language model. "
        "When presented with a user request, your task is to craft a well-structured, "
        "privacy-preserving prompt for the language model. "
        "Focus on abstracting specific details from the user's query while capturing the essence of the request. "
        "Ensure that the prompt is general enough to avoid any personally identifiable information "
        "while still allowing the language model to generate useful and relevant responses. "
        "After creating the prompt, provide it directly without any additional commentary. "
        "Do not attempt to complete the user's original request; your sole responsibility is to present the generated prompt."
    )

    return dspy.make_signature(
        fields, instruction, "PromptCreaterSignature"
    )

PromptCreaterSignature = make_prompt_creater_signature()

def make_info_aggregator_signature():
    """
    You are a helpful assistant. Respond to queries from the user.
    """

    fields = {
        "current_user_message": (str, dspy.InputField(desc="The user's request to be fulfilled.")),
        "model_example_responses": (str, dspy.InputField(desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information.")),
        "further_information": (
            str,
            dspy.OutputField(),
        )
    }

    instruction = ("You are a highly skilled assistant capable of generating engaging and innovative responses. "
                   "Analyze the user's request thoroughly and provide a detailed response that not only answers their query "
                   "but also showcases creativity and understanding of the context. Your output should include well-structured content, "
                   "relevant examples, and an engaging tone appropriate for the task at hand. "
                   "Always aim to surprise the user with insightful perspectives that go beyond basic information, "
                   "ensuring clarity and emotional resonance in your communication.")

    return dspy.make_signature(
        fields, instruction, "InfoAggregatorSignature"
    )

InfoAggregatorSignature = make_info_aggregator_signature()

class Polish(dspy.Module):
    def __init__(self):
        self.prompt_creater = dspy.ChainOfThought(PromptCreaterSignature)
        self.info_aggregator = dspy.Predict(InfoAggregatorSignature)
        self.untrusted_model = untrusted_lm
        self.token_ratios: dict[str, float] = {
            "current_user_message": 2 / 15,
            "conversation_history": 2 / 15,
            "conversation_summary": 1 / 15,
            "tool_history": 5 / 15,
            "tool_summary": 1 / 15,
        }

    def get_token_limits(self) -> dict[str, int]:
        return token_limit_ratio_to_count(
            self.token_ratios, len(get_template(self.prompt_creater))
        )


    def forward(
            self,
            current_user_message: str,
            conversation_memory: ConversationMemory,
            tool_memory: ToolMemory,
    ):

        rewrite_inputs = dict(
            current_user_message=current_user_message,
            conversation_history=conversation_memory.history_str(),
            conversation_summary=conversation_memory.summary,
            tool_history=tool_memory.history_str(),
            tool_summary=tool_memory.summary,
        )
        rewrite_inputs = truncate_tokens_all(
            rewrite_inputs, self.get_token_limits()
        )

        try:
            prompt = self.prompt_creater(**rewrite_inputs).created_prompt
            response = self.untrusted_model(prompt)[0]
            output = self.info_aggregator(current_user_message=current_user_message, model_example_responses=response)
        except Exception:
            return dspy.Prediction(prompt="", output="", gptResponse="")

        return dspy.Prediction(prompt=prompt, output=output.further_information, gptResponse=response)


# polisher = Polish()
#loaded_papillon.load('./optimized_prompts/llama_31_8b_instruct_prompt.json')

# while True:
#     user_query = input("Your Query > ")
#     pred = polisher(user_query)
#     print("PAPILLON PROMPT > ", pred.prompt)
#     print("PAPILLON OUTPUT > ", pred.output)