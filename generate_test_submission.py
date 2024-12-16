import os
import json
import pickle
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoProcessor
from dataclasses import dataclass
from models.phi35.instructcir_phi35_v import InstructCIRLlavaPhi35ForConditionalGeneration

from dataset.validate_datasets import CIRRDataset, CIRCODataset


device = "cuda:0"


@dataclass
class ValidateCollator(object):
    processor: transformers.ProcessorMixin = None
    mode: str = "classic"

    def __call__(self, instances):
        if self.mode == "classic":
            images = [instance["image"] for instance in instances]
            image_names = [instance["image_name"] for instance in instances]
            image_prompt = "<image>\n Describe this image in one word:"
            prompt_message = {
                'role': 'user',
                'content': image_prompt,
            }
            prompt = self.processor.tokenizer.apply_chat_template(
                [prompt_message], tokenize=False, add_generation_prompt=True
            )
            prompt = prompt[3:]
            input_texts = [prompt] * len(images)
            inputs = self.processor(input_texts, images, return_tensors="pt", padding=True)
            image_names = torch.utils.data.dataloader.default_collate(image_names)
            batch = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs["pixel_values"],
                "image_name": image_names,
            }
            return batch
        else:
            images = [instance["reference_image"] for instance in instances]
            prompt_template = "<image>Modify this image with \"{}\", desribe modified image in one word:"

            input_texts = []
            for instance in instances:
                prompt = prompt_template.format(instance["relative_caption"])
                prompt_message = {
                    'role': 'user',
                    'content': prompt,
                }
                prompt = self.processor.tokenizer.apply_chat_template(
                    [prompt_message], tokenize=False, add_generation_prompt=True
                )
                prompt = prompt[3:]
                input_texts.append(prompt)
            inputs = self.processor(input_texts, images, return_tensors="pt", padding=True)

            batch = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs["pixel_values"],
            }

            if "query_id" in instances[0].keys():
                query_ids = [instance["query_id"] for instance in instances]
                query_ids = torch.utils.data.dataloader.default_collate(query_ids)
                batch["query_id"] = query_ids
            if "group_members" in instances[0].keys():
                group_members = [instance["group_members"] for instance in instances]
                group_members = torch.utils.data.dataloader.default_collate(group_members)
                batch["group_members"] = group_members
            if "pair_id" in instances[0].keys():
                pair_ids = [instance["pair_id"] for instance in instances]
                pair_ids = torch.utils.data.dataloader.default_collate(pair_ids)
                batch["pair_id"] = pair_ids
            if "reference_name" in instances[0].keys():
                reference_names = [instance["reference_name"] for instance in instances]
                reference_names = torch.utils.data.dataloader.default_collate(reference_names)
                batch["reference_name"] = reference_names

            return batch


@torch.no_grad()
def extract_image_features(dataset, model, preprocess, batch_size = 64, num_workers = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    collator = ValidateCollator(processor=preprocess, mode="classic")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collator)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        names = batch.get('image_name')
        if names is None:
            names = batch.get('reference_name')

        with torch.no_grad():
            batch_features = model.encode(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                pixel_values=batch.get("pixel_values", None).to(device),
                output_hidden_states=True, 
                return_dict=True
            )
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path, model, preprocess, submission_name) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    #clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    #clip_model = clip_model.float().eval()

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess=None)
    index_features, index_names = extract_image_features(classic_test_dataset, model, preprocess)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess=None)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, model, index_features, index_names, preprocess)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = os.path.join('./submission', 'cirr')
    os.makedirs(submissions_folder_path, exist_ok=True)

    with open(os.path.join(submissions_folder_path, f"{submission_name}.json"), 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(os.path.join(submissions_folder_path, f"subset_{submission_name}.json"), 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset, model, index_features, index_names, preprocess):
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(model, relative_test_dataset, preprocess)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features.float() @ index_features.T.float()
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(model, relative_test_dataset, preprocess) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    collator = ValidateCollator(processor=preprocess, mode="relative")
    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10, pin_memory=False, collate_fn=collator)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        group_members = batch['group_members']

        group_members = np.array(group_members).T.tolist()

        predicted_features = model.encode(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            pixel_values=batch.get("pixel_values", None).to(device),
            output_hidden_states=True, 
            return_dict=True
        )

        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path, model, preprocess, submission_name) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    #clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    #clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess=None)
    index_features, index_names = extract_image_features(classic_test_dataset, model, preprocess)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess=None)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, model, index_features, index_names, preprocess)

    submissions_folder_path = os.path.join('./submission', 'circo')
    os.makedirs(submissions_folder_path, exist_ok=True)

    with open(os.path.join(submissions_folder_path, f"{submission_name}.json"), 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(model, relative_test_dataset, preprocess):
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    collator = ValidateCollator(processor=preprocess, mode="relative")
    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                    pin_memory=False, collate_fn=collator, shuffle=False)

    predicted_features_list = []
    query_ids_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        query_ids = batch['query_id']

        predicted_features = model.encode(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            pixel_values=batch.get("pixel_values", None).to(device),
            output_hidden_states=True, 
            return_dict=True
        )

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset, model, index_features, index_names, preprocess) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids = circo_generate_test_predictions(model, relative_test_dataset, preprocess)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features.float() @ index_features.T.float()
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, help="cirr_results", default="cirr_results")
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", type=str, choices=['cirr', 'circo'], help="Dataset to use", default="cirr")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", default="/home/wlzhong/dataset/cirr")
    parser.add_argument("--model_name_or_path", type=str, default="uta-smile/instructcir_llava_phi35_clip224_lp")

    args = parser.parse_args()

    kwargs = {"device_map": "cuda"}
    kwargs["device_map"] = {"": device}
    kwargs["torch_dtype"] = torch.float16
    kwargs["_attn_implementation"] = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = InstructCIRLlavaPhi35ForConditionalGeneration.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, **kwargs)

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, model, processor, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path,  model, processor, args.submission_name)
    else:
        raise ValueError("Dataset not supported yet!")


if __name__ == '__main__':
    main()