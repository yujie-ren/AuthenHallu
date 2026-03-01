from datasets import load_dataset
import json


MODEL_MAP = {
    "Llama_3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama_3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma_3": "google/gemma-3-27b-it",
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen_3": "Qwen/Qwen3-32B",
    "Qwen_2.5": "Qwen/Qwen2.5-7B-Instruct"
}

scheme1 = ("Gemma_3", "Llama_3.1", "Llama_3.1")
scheme2 = ("Gemma_3", "Mistral_7B", "Llama_3.3")
scheme3 = ("Gemma_3", "Mistral_7B", "Qwen_3", "Llama_3.1", "Llama_3.3")


def main():
    ensemble_scheme = scheme3

    ###############################################
    # Load dataset
    ###############################################
    # load predicitions from individual models
    ds_list = []
    for model_name in ensemble_scheme:
        path = "annotation_extraction/results_v2/category/" + "results_" + model_name + "_category_v2.json"
        ds = load_dataset("json", data_files=path, split="train")
        ds_list.append(ds)

    ###############################################
    # Ensemble via majority voting
    ###############################################
    example_list = []
    for ex_tuple in zip(*ds_list):
        # check consistency
        entry_id_list = [ex["entry_id"] for ex in ex_tuple]
        assert len(set(entry_id_list)) == 1, "not consistent"

        example = {}
        if ex_tuple[0].get("prompt1"):  # prompt1 exists
            # check consistency
            prompt1_list = [ex["prompt1"] for ex in ex_tuple]
            assert len(set(prompt1_list)) == 1, "not consistent"
            answer1_list = [ex["answer1"] for ex in ex_tuple]
            assert len(set(answer1_list)) == 1, "not consistent"

            # majority voting 
            pred1_list = [ex["pred1"].strip() for ex in ex_tuple]  # extract labels
            pred1_majority = majority_voting(pred1_list)

            # save
            example["entry_id"] = entry_id_list[0]
            example["prompt1"] = prompt1_list[0]
            example["answer1"] = answer1_list[0]
            example["pred1"] = pred1_majority
        
        if ex_tuple[0].get("prompt2"):  # prompt2 exists
            # check consistency
            prompt2_list = [ex["prompt2"] for ex in ex_tuple]
            assert len(set(prompt2_list)) == 1, "not consistent"
            answer2_list = [ex["answer2"] for ex in ex_tuple]
            assert len(set(answer2_list)) == 1, "not consistent"

            # majority voting
            pred2_list = [ex["pred2"].strip() for ex in ex_tuple]
            pred2_majority = majority_voting(pred2_list)

            # save
            example["entry_id"] = entry_id_list[0]
            example["prompt2"] = prompt2_list[0]
            example["answer2"] = answer2_list[0]
            example["pred2"] = pred2_majority
        
        # save to list
        example_list.append(example)

    ###############################################
    # Save results of majority voting to json
    ###############################################
    assert len(example_list) == 163, "should be 163 dialogues"
    file_name = "-".join(ensemble_scheme)
    file_path = f"subsequent_experiment/ensemble_categorization/results/{file_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(example_list, f, ensure_ascii=False, indent=4)



def majority_voting(pred_list: list[str]):

    # count for each label
    vote_counts = {}
    for pred in pred_list:
        vote_counts[pred] = vote_counts.get(pred, 0) + 1
    # find the max vote
    max_votes = max(vote_counts.values())
    # check winners
    winners = [label for label, votes in vote_counts.items() if votes == max_votes]

    # majority voting
    if len(winners) == 1:
        pred_majority = winners[0]
    else:
        pred_majority = pred_list[0]  # if tie, follow Gemma3
    return pred_majority



if __name__ == "__main__":
    main()