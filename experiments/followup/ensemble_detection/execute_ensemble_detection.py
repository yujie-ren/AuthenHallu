from datasets import load_dataset
import json
from sklearn.metrics import classification_report, confusion_matrix


MODEL_MAP = {
    "Llama_3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama_3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma_3": "google/gemma-3-27b-it",
    "Mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen_3": "Qwen/Qwen3-32B",
    "Qwen_2.5": "Qwen/Qwen2.5-7B-Instruct"
}

scheme1 = ("Gemma_3", "Qwen_3", "Llama_3.1")
scheme2 = ("Gemma_3", "Qwen_2.5", "Qwen_3")
scheme3 = ("Gemma_3", "Qwen_2.5", "Qwen_3", "Llama_3.1", "Llama_3.3")


def main():
    ensemble_scheme = scheme3

    ###############################################
    # Load dataset
    ###############################################
    # load predicitions from individual models
    ds_list = []
    for model_name in ensemble_scheme:
        path = ""
        ds = load_dataset("json", data_files=path, split="train")
        ds_list.append(ds)

    ###############################################
    # Ensemble via majority voting
    ###############################################
    example_list = []
    for ex_tuple in zip(*ds_list):
        # check consistency
        entry_id_list = [sample['entry_id'] for sample in ex_tuple]
        assert len(set(entry_id_list)) == 1, "not consistent!"
        prompt1_list = [sample['prompt1'] for sample in ex_tuple]
        assert len(set(prompt1_list)) == 1, "not consistent!"
        answer1_list = [sample['answer1'] for sample in ex_tuple]
        assert len(set(answer1_list)) == 1, "not consistent!"
        prompt2_list = [sample['prompt2'] for sample in ex_tuple]
        assert len(set(prompt2_list)) == 1, "not consistent!"
        answer2_list = [sample['answer2'] for sample in ex_tuple]
        assert len(set(answer2_list)) == 1, "not consistent!"

        # extract labels of turn1 & turn2
        pred1_list = [sample['pred1'].strip() for sample in ex_tuple]
        pred2_list = [sample['pred2'].strip() for sample in ex_tuple]

        # majority voting
        pred1_majority = max(["Hallucination", "No Hallucination"], key=pred1_list.count)
        pred2_majority = max(["Hallucination", "No Hallucination"], key=pred2_list.count)

        # save result
        example = {}
        example["entry_id"] = entry_id_list[0]
        example["prompt1"] = prompt1_list[0]
        example["answer1"] = answer1_list[0]
        example["pred1"] = pred1_majority
        example["prompt2"] = prompt2_list[0]
        example["answer2"] = answer2_list[0]
        example["pred2"] = pred2_majority

        example_list.append(example)

    ###############################################
    # Save results of majority voting to json
    ###############################################
    assert len(example_list) == 400, "should be 400 dialogues"
    file_name = "-".join(ensemble_scheme)
    file_path = f"subsequent_experiment/ensemble_detection/results/{file_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(example_list, f, ensure_ascii=False, indent=4)


    ###############################################
    # Compute metrics
    ###############################################
    # load ground truth
    path = "annotation_extraction/ground_truth_400.json"
    ds_groundTruth = load_dataset("json", data_files=path, split="train")

    # check consistency
    for ex_majority, ex_groundTruth in zip(example_list, ds_groundTruth):
        assert ex_majority["entry_id"] == str(ex_groundTruth["entry_id"]), "Not consistent!"
        assert ex_majority["prompt1"] == str(ex_groundTruth["prompt1"]), "Not consistent!"
        assert ex_majority["answer1"] == str(ex_groundTruth["answer1"]), "Not consistent!"
        assert ex_majority["prompt2"] == str(ex_groundTruth["prompt2"]), "Not consistent!"
        assert ex_majority["answer2"] == str(ex_groundTruth["answer2"]), "Not consistent!"

    # extract labels
    pred_turn1 = [example["pred1"].strip() for example in example_list]
    pred_turn2 = [example["pred2"].strip() for example in example_list]
    groundTruth_turn1 = [example["label1"].strip() for example in ds_groundTruth]
    groundTruth_turn2 = [example["label2"].strip() for example in ds_groundTruth]

    pred_list = pred_turn1 + pred_turn2
    groundTruth_list = groundTruth_turn1 + groundTruth_turn2

    # compute classification report & confusion matrix
    cr = classification_report(groundTruth_list, pred_list, output_dict=True, zero_division=0)
    cm = confusion_matrix(groundTruth_list, pred_list, labels=["Hallucination", "No Hallucination"])
    
    # save to json file
    result = {
        "Classification_Report": cr,
        "Confusion_Matrix_(Hallu_NoHallu)": cm.tolist()
    }
    path = f"subsequent_experiment/ensemble_detection/results/{file_name}_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()