import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

def verify(user_query,model_answer, model, processor):

    ##################################################
    # Step 2: Construct inputs
    ##################################################
    system_prompt = "You are an expert in hallucination categorization. Answer only with 'A', 'B' or 'C'."
    user_prompt = (
    "A hallucination can be categorized into one of the three categories:.\n"
    "Input-conflicting hallucinations (A) appear when generated content differs from what was given to the model as source (the model does not answer the question).\n" 
    "Context-conflicting hallucinations (B) appear as information that is out of place and conflicts with what was previously generated (the model contradicts itself).\n" 
    "Fact-conflicting hallucinations (C) is content that is not factual nor faithful to what is known to be true and not based on any knowledge (the model produces unfactual content).\n"
    "Your task is to detect which category matches the given hallucination in the generated answer.\n"
    "Respond strictly and only with one of the following labels:\n"
    "- A\n"
    "- B\n"
    "- C\n\n"
    f"User Query: {user_query}\n"
    f"Generated Answer: {model_answer}\n"
    "Category:"
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
        
    inputs = processor.apply_chat_template(**kwargs)
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    ##################################################
    # Step 3: Get outputs
    ##################################################
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
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
    
    json_file = "ground_truth_category_v2.json"  # Output file for saving the processed summaries
    # Loading the processed File
    print(f"Main: Loading the File {json_file}...\n")
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)  

    print(f"\nMain: Loaded\n")
    # === Checkpoint loading ===
    results_file = f"results_{model_name}_category_v2.json"
    results = []
    completed_ids = set()

    if os.path.exists(results_file):
        print(f"\nMain: Found existing results file {results_file}, loading...\n")
        with open(results_file, "r", encoding="utf-8") as file:
            results = json.load(file)
        completed_ids = {entry["entry_id"] for entry in results}
        print(f"Main: Resuming from last saved state — {len(completed_ids)} entries already processed.\n")
    else:
        print(f"\nMain: No existing results file found. Starting fresh.\n")

    save_interval = 40
    new_results = []

    for idx, entry in enumerate(data):
        entry_id = entry["ID"]

        if entry_id in completed_ids:
            print(f"Skipping entry {entry_id}, already processed.")
            continue

        result = {"entry_id": entry_id}

        # Turn 1
        if entry.get("label1") == "Hallucination":
            print(f"Verifying Category for Entry {entry_id} - Turn 1...")
            prompt1 = entry["prompt1"]
            answer1 = entry["answer1"]
            pred1 = verify(prompt1, answer1, model, processor)
            print("~Verifier Output Prompt 1~:\n", pred1)

            result.update({
                "prompt1": prompt1,
                "answer1": answer1,
                "pred1": pred1
            })

        # Turn 2
        if entry.get("label2") == "Hallucination":
            print(f"Verifying Category for Entry {entry_id} - Turn 2...")
            prompt2 = entry["prompt2"]
            answer2 = entry["answer2"]
            pred2 = verify(prompt2, answer2, model, processor)
            print("~Verifier Output Prompt 2~:\n", pred2)

            result.update({
                "prompt2": prompt2,
                "answer2": answer2,
                "pred2": pred2
            })

        if "pred1" in result or "pred2" in result:
            results.append(result)
            new_results.append(result)

        # === Save checkpoint every N entries ===
        if len(new_results) >= save_interval:
            print(f"\nMain: Saving checkpoint after {len(new_results)} new entries...\n")
            with open(results_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4, ensure_ascii=False)
            new_results = []

    # === Final Save ===
    print(f"\nMain: Final save of all results...\n")
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"\nMain: Results saved in {results_file}\n")