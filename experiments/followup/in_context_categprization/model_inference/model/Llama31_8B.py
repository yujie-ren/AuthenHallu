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
        model_name = "Llama31_8B"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_name],
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
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Ensure pad token is set
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    ##################################################
    # Step 3: Get outputs
    ##################################################
    outputs = model.generate(
        **inputs,
        max_new_tokens=6,  # 6
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_ids=inputs["input_ids"]

    ##################################################
    # Step 4: Skip input ids & Decoding
    ##################################################
    output_ids = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text


if __name__ == "__main__":
    inference()