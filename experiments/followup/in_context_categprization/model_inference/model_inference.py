import argparse
import importlib
from datasets import load_dataset
import json


MODEL_MAP = {
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma3_27B": "google/gemma-3-27b-it",
    "Qwen25_7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3_32B": "Qwen/Qwen3-32B",
    "Llama31_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama33_70B": "meta-llama/Llama-3.3-70B-Instruct",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Llama33_70B")
    args = parser.parse_args()
    model_name = args.model_name

    print("==================================")
    print("***** model inference *****")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("==================================")

    #####################################
    # Step 1: Load dataset
    #####################################
    path = ""
    ds_groundTruth = load_dataset("json", data_files=path, split="train")
    # print(ds_groundTruth)

    result = []
    i = 0
    for example in ds_groundTruth:
        if example["label2"] != "Hallucination":
            continue
        assert example["label2"] == "Hallucination"
        i = i + 1
        print(i)
        user_query1 = example["prompt1"]
        model_answer1 = example["answer1"]
        user_query2 = example["prompt2"]
        model_answer2 = example["answer2"]

        #####################################
        # Step 2: Prepare inputs
        #####################################
        system_prompt = "You are an expert in hallucination categorization. Answer only with 'A', 'B' or 'C'."
        user_prompt = (
        "A hallucination can be categorized into one of the three categories:.\n"
        "Input-conflicting hallucinations (A) appear when generated content differs from what was given to the model as source (the model does not answer the question).\n" 
        "Context-conflicting hallucinations (B) appear as information that is out of place and conflicts with what was previously generated (the model contradicts itself).\n" 
        "Fact-conflicting hallucinations (C) is content that is not factual nor faithful to what is known to be true and not based on any knowledge (the model produces unfactual content).\n"
        "Given a conversation consisting of two Query-Answer pairs, it is known that the second answer (Generated Answer 2) contains hallucinations. "
        "Your task is to detect which category matches the given hallucination in the second answer (Generated Answer 2).\n"
        "Respond strictly and only with one of the following labels:\n"
        "- A\n"
        "- B\n"
        "- C\n\n"
        "Conversation:\n"
        f"User Query 1: {user_query1}\n"
        f"Generated Answer 1: {model_answer1}\n"
        f"User Query 2: {user_query2}\n"
        f"Generated Answer 2: {model_answer2}\n"
        "Category:"
        )

        # system_prompt = "You are an AI assistant"
        # user_prompt = "Who are you?"

        #####################################
        # Step 3: Get outputs
        #####################################
        model_module = importlib.import_module(f"model.{model_name}")
        output_text = model_module.inference(system_prompt=system_prompt, user_prompt=user_prompt)
        print(output_text)
        example["pred_category2"] = output_text
        result.append(example)

    #####################################
    # Step 4: Save outputs
    #####################################
    path = f"/{model_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    main()