import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        model_name = "Qwen25_7B"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_name],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
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
    messages=[{"role": "system", "content": system_prompt,}, {"role": "user", "content": user_prompt },]
    kwargs = {
        "conversation": messages,
        "add_generation_prompt": True,
        "return_tensors": "pt"
    }
    input_ids = tokenizer.apply_chat_template(**kwargs).to(model.device)

    ##################################################
    # Step 3: Get outputs
    ##################################################
    outputs = model.generate(
        input_ids,
        max_new_tokens=6,  # 6
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    ##################################################
    # Step 4: Skip input ids & Decoding
    ##################################################
    output_ids = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text


if __name__ == "__main__":
    inference()