import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import argparse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
from huggingface_hub import login
login("")  # safer if stored in an environment variable


MODEL_MAP = {
    "Gemma_3": "google/gemma-3-27b-it",
}

def verify(user_query,model_answer, model, processor):

    ##################################################
    # Step 2: Construct inputs
    ##################################################
    system_prompt = "You are a specialist in identifying hallucinations. Respond solely with either 'Hallucination' or 'No Hallucination'."
    user_prompt = (
    "A hallucination is content generated that contradicts the input, the model’s own context, or established facts.\n"
    "Your task is to determine whether the answer provided to the user prompt is a hallucination.\n"
    "Respond exclusively with one of these labels:\n"
    "- Hallucination\n"
    "- No Hallucination\n\n"
    f"User Query: {user_query}\n"
    f"Generated Answer: {model_answer}\n"
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
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id  # Accessing the internal tokenizer
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
    parser.add_argument("--model_name", type=str, default="Mistral_7B")
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
        
        if user_prompt_1:
            print(f"Verifying Factuality for Prompt 1...")
            
            verification_result_1 = verify(user_prompt_1, model_answer_1, model, processor)
        else:
            verification_result_1 = ""
        print("~Verifier Output Prompt 1~:\n", verification_result_1)

        if user_prompt_2:
            print(f"\nVerifying Factuality for Prompt 2...")
            
            verification_result_2 = verify(user_prompt_2, model_answer_2, model, processor)
        else:
            verification_result_2 = ""
        print("~Verifier Output Prompt 2~:\n", verification_result_2)
        
        # Store the results
        results.append({
            "entry_id": entry_id,
            "prompt1": user_prompt_1,
            "answer1": model_answer_1,
            "pred1": verification_result_1,
            "prompt2": user_prompt_2,
            "answer2": model_answer_2,
            "pred2": verification_result_2,
        })
        #print(results)
    
    print(f"\nMain: All results will be saved...\n")
    
    # Save results to a new JSON file
    with open(f"results_{model_name}_subset_pp_v2.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    
    print(f"\nMain: Results saved in {file}\n")
    # ======================================================================