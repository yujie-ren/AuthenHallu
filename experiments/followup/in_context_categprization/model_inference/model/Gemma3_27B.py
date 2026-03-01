from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

MODEL_MAP = {
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma3_27B": "google/gemma-3-27b-it",
    "Qwen25_7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3_32B": "Qwen/Qwen3-32B",
    "Llama31_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama33_70B": "meta-llama/Llama-3.3-70B-Instruct",
}

model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        model_name = "Gemma3_27B"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_MAP[model_name], 
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_MAP[model_name])
    return model, processor


def inference(
    system_prompt: str = "You are an AI assistant",
    user_prompt: str = "Who are you?"
    ):

    model, processor = load_model()
    # model_name = "Gemma3_27B"

    ###########################################
    # Step2: Prepare inputs
    ###########################################
    # system_prompt = "You are an AI assistant"
    # user_prompt = "Who are you?"
    messages=[
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    kwargs = {
        "conversation": messages,  
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True
    }
    inputs = processor.apply_chat_template(**kwargs).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    ##################################################
    # Step 3: Get outputs
    ##################################################
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,  # 6
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id  # Accessing the internal tokenizer
        )

    ##################################################
    # Step 4: Skip input ids & Decoding
    ##################################################
    generation = outputs[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded


if __name__ == "__main__":
    inference()