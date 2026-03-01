import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def verify(user_query,model_answer, model_name, tokenizer, model):

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
            print("Else chosen...")
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
    
    json_file = "ground_truth_category_v2.json"  # Output file for saving the processed summaries
    # Loading the processed File
    print(f"Main: Loading the File {json_file}...\n")
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)  

    print(f"\nMain: Loaded\n")
    results = []

    for entry in data:
        entry_id = entry["ID"]
        result = {"entry_id": entry_id}

        # Turn 1
        if entry.get("label1") == "Hallucination":
            print(f"Verifying Category for Entry {entry_id} - Turn 1...")
            prompt1 = entry["prompt1"]
            answer1 = entry["answer1"]
            pred1 = verify(prompt1, answer1, model_name, tokenizer, model)
            print("~Verifier Output Prompt 1~:\n", pred1)

            result.update({
                "prompt1": prompt1,
                "answer1": answer1,
                "pred1": pred1
            })

        # Turn 2
        if entry.get("label2") == "Hallucination":
            print(f"Verifying Factuality for Entry {entry_id} - Turn 2...")
            prompt2 = entry["prompt2"]
            answer2 = entry["answer2"]
            pred2 = verify(prompt2, answer2, model_name, tokenizer, model)
            print("~Verifier Output Prompt 2~:\n", pred2)

            result.update({
                "prompt2": prompt2,
                "answer2": answer2,
                "pred2": pred2
            })

        # Only add to results if at least one hallucinated turn is found
        if "pred1" in result or "pred2" in result:
            results.append(result)

    
    print(f"\nMain: All results will be saved...\n")
    
    # Save results to a new JSON file
    with open(f"results_{model_name}_category_v2.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    
    print(f"\nMain: Results saved in {file}\n")
    # ======================================================================