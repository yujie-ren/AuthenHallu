import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
from huggingface_hub import login
login("")  # safer if stored in an environment variable
from huggingface_hub import snapshot_download


MODEL_MAP = {
    "Llama_3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama_3.3": "meta-llama/Llama-3.3-70B-Instruct",
    #"Gemma_3": "google/gemma-3-27b-it",
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen_3": "Qwen/Qwen3-32B",
    "Qwen_2.5": "Qwen/Qwen2.5-7B-Instruct"
}

def verify(user_query1, model_answer1, user_query2, model_answer2, model_name, tokenizer, model):

    ##################################################
    # Step 2: Construct inputs
    ##################################################
    system_prompt = "You are an expert in hallucination detection. Answer only with 'Hallucination' or 'No Hallucination'."
    user_prompt = (
    "A hallucination refers to generated content that conflicts with the input, the model's own context, or known facts.\n"
    "Your task is to detect if the second answer (Generated Answer 2) given to the second user prompt (User Query 2) is a hallucination "
    "based on the complete conversation, including the context from User Query 1 and Generated Answer 1.\n"
    "Respond strictly and only with one of the following labels:\n"
    "- Hallucination\n"
    "- No Hallucination\n\n"
    "Conversation:\n"
    f"User Query 1: {user_query1}\n"
    f"Generated Answer 1: {model_answer1}\n"
    f"User Query 2: {user_query2}\n"
    f"Generated Answer 2: {model_answer2}\n"
    "Label:"
    )

    messages=[
        {"role": "system", "content": system_prompt,
        },
        {"role": "user", "content": user_prompt
        },
    ]
    
    if model_name == "Mistral_7B":
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_inference.generate import generate
        

        request = ChatCompletionRequest(messages=[
            UserMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ])

        tokens = tokenizer.encode_chat_completion(request).tokens
        out_tokens, _ = generate(
            [tokens],
            model,
            max_tokens=6,
            temperature=0.0,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        output_text = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        
        print("---------")
        print(output_text)
        print("---------")
        return output_text
        

    else:
        if model_name == "Qwen_3":
            kwargs = {
                "conversation": messages,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "enable_thinking": False
            }
            print("No Think Mode!")
            input_ids = tokenizer.apply_chat_template(**kwargs).to(model.device)

        elif model_name == "Llama_3.3" or model_name == "Llama_3.1":
            # LLaMA 3.3 requires apply_chat_template + separate tokenization
            print("Padding and Truncation activated!")

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Ensure pad token is set
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

        else:
            kwargs = {
                "conversation": messages,
                "add_generation_prompt": True,
                "return_tensors": "pt"
            }
            input_ids = tokenizer.apply_chat_template(**kwargs).to(model.device)

    ##################################################
    # Step 3: Get outputs
    ##################################################
    if (model_name =="Llama_3.3" or model_name == "Llama_3.1") :
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_ids=inputs["input_ids"]
    
    else:
        outputs = model.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    ##################################################
    # Step 4: Skip input ids & Decoding
    ##################################################
    output_ids = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("---------")
    print(output_text)
    print("---------")
    return output_text


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama_3.1")
    args = parser.parse_args()
    model_name = args.model_name
    print("===============================")
    print("***** Model Inference *****")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===============================")

    ##################################################
    # Step 1: Load tokenizer & model
    ##################################################
    if model_name == "Mistral_7B":
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        tokenizer = MistralTokenizer.from_file(f".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/tokenizer.model.v3")
        from mistral_inference.transformer import Transformer
        model = Transformer.from_folder(".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3")
    
    elif model_name =="Llama_3.3":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        load_in_8bit=True,
        device_map="auto",
        )

    elif model_name =="Llama_3.1":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        device_map="auto",
        )
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )
    
    print(f"Loaded model {model_name}")
    
    json_file = "subset_conversations_v2.json"  # Output file for saving the processed summaries
    # Loading the processed File
    print(f"Main: Loading the File {json_file}...\n")
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)  

    print(f"\nMain: Loaded\n")
    results = []
    
    for entry in data:
        
        # Extract original prompts and answers
        entry_id = entry["id"]
        user_prompt_1 = entry["prompt1"]
        model_answer_1 =entry["answer1"]
        user_prompt_2 = entry["prompt2"]
        model_answer_2 =entry["answer2"]   
        
        
        print("\nMain: Verify Factuality...\n")
        
        if (user_prompt_1 and user_prompt_2):
            print(f"Verifying Factuality for Prompt 1...")
            
            verification_result = verify(user_prompt_1, model_answer_1, user_prompt_2, model_answer_2, model_name, tokenizer, model)
        else:
            verification_result = ""
        print("~Verifier Output Prompt 1~:\n", verification_result)
        
        # Store the results
        results.append({
            "entry_id": entry_id,
            "prompt1": user_prompt_1,
            "answer1": model_answer_1,
            "prompt2": user_prompt_2,
            "answer2": model_answer_2,
            "pred": verification_result,
        })
        #print(results)
    
    print(f"\nMain: All results will be saved...\n")
    
    # Save results to a new JSON file
    with open(f"results_{model_name}_subset_context_v2.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    
    print(f"\nMain: Results saved in {file}\n")
    # ======================================================================