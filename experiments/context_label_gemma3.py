import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
import argparse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
from huggingface_hub import login
login("")  # safer if stored in an environment variable


MODEL_MAP = {
    #"Llama3.3_70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma_3": "google/gemma-3-27b-it",
    #"Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    #"Qwen_3": "Qwen/Qwen3-32B"
}

def verify(user_query1 ,model_answer1, user_query2, model_answer2, model, processor):

    ##################################################
    # Step 2: Construct inputs
    ##################################################
    system_prompt = "You are an expert in context detection. Answer only with 'Yes' or 'No'."
    user_prompt = (
    "A conversation in which both turns talk about spain are contextually connected, while a conversation in which the first turn " 
    "talks about cake while the second turn discusses business e-mails is not contextually connected.\n"
    "Your task is to detect which conversations are contextually connected between both turns.\n"
    "Respond strictly and only with one of the following labels:\n"
    "- Yes\n"
    "- No\n"
    "Yes, if the conversation is contextually connecetd between both turns, and No, "
    "if they are two complete independant turns.\n\n"
    "Turn 1:\n"
    f"User Query: {user_query1}\n"
    f"Generated Answer: {model_answer1}\n"
    "Turn 2:\n"
    f"User Query: {user_query2}\n"
    f"Generated Answer: {model_answer2}\n"
    "Label:"
    )

    messages=[
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]
        },
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]
        },
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
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    ##################################################
    # Step 4: Skip input ids & Decoding
    ##################################################
    generation = outputs[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    print("---------")
    print(decoded)
    print("---------")
    return decoded


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Gemma_3")
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
    
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_MAP[model_name], 
        device_map="auto",
        ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_MAP[model_name])
    
    
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

        if user_prompt_1 and user_prompt_2:
            print(f"Verifying Factuality for Prompt 1...")
            
            verification_result_1 = verify(user_prompt_1, model_answer_1, user_prompt_2, model_answer_2, model, processor)
        else:
            verification_result_1 = ""
        print("~Verifier Output Prompt 1~:\n", verification_result_1)
        
        # Store the results
        results.append({
            "entry_id": entry_id,
            "prompt1": user_prompt_1,
            "answer1": model_answer_1,
            "prompt2": user_prompt_2,
            "answer2": model_answer_2,
            "pred": verification_result_1,
        })
        #print(results)
    
    print(f"\nMain: All results will be saved...\n")
    
    # Save results to a new JSON file
    with open(f"results_{model_name}_context_v2.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    
    print(f"\nMain: Results saved in {file}\n")
    # ======================================================================