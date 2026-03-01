from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.transformer import Transformer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate

MODEL_MAP = {
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma3_27B": "google/gemma-3-27b-it",
    "Qwen25_7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3_32B": "Qwen/Qwen3-32B",
    "Llama31_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama33_70B": "meta-llama/Llama-3.3-70B-Instruct",
}

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        cache_path = "/work/bbe2549/.cache/mistral_models/7B-Instruct-v0.3"
        tokenizer = MistralTokenizer.from_file(f"{cache_path}/tokenizer.model.v3")
        model = Transformer.from_folder(cache_path)
    return model, tokenizer

def inference(
    system_prompt: str = "You are an AI assistant",
    user_prompt: str = "Who are you?"
    ):
    model, tokenizer = load_model()

    ###########################################
    # Step2: Prepare inputs 
    ###########################################
    # system_prompt = "You are an AI assistant"
    # user_prompt = "Who are you?"
    completion_request = ChatCompletionRequest(messages=[SystemMessage(content=system_prompt), UserMessage(content=user_prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    ###########################################
    # Step3: Get output 
    ###########################################
    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=6,  # 6
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )
    output_text = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    return output_text


if __name__ == "__main__":
    inference()