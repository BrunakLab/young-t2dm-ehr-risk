import sys
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from lightning.fabric import is_wrapped
from tqdm import tqdm

import diabnet.learn.train as train
from diabnet.datasets.patient_history import load_required_indexes
from diabnet.models.binned_trajectory_risk_model import BinnedTrajectoryRiskModel
from diabnet.models.single_registry_trajectory_risk_model import SingleRegistryTrajectoryRiskModel
from diabnet.utils.learn import get_dataset_loader
from diabnet.utils.vocab import PAD_CODE, CodeParser


def load_interaction_partners(path) -> List[Tuple[str, str]]:
    codes = open(path, "r").read().splitlines()
    interaction = []
    for code_modality in codes:
        token, modality = code_modality.split("\t")
        interaction.append((int(token), modality))
    return interaction


def get_attributable_layers(model, modalities):
    if isinstance(model, SingleRegistryTrajectoryRiskModel):
        code_embeddings_layers = [model.embedder.code_embedder]
        if hasattr(model.embedder, "age_embedder"):
            age_embeddings_layers = [
                model.embedder.age_embedder.add_fc,
                model.embedder.age_embedder.scale_fc,
            ]
        else:
            age_embeddings_layers = None

    elif isinstance(model, BinnedTrajectoryRiskModel):
        code_embeddings_layers, age_embeddings_layers = [], []
        for modality in modalities:
            code_embeddings_layers.append(model.embedders[modality].code_embed)

        if hasattr(model, "age_embedder"):
            age_embeddings_layers.extend(
                [
                    model.age_embedder.add_fc,
                    model.age_embedder.scale_fc,
                ]
            )
        else:
            age_embeddings_layers = None
    else:
        print(f"Model of type: {type(model)} cannot be attributed")
        sys.exit(1)

    return code_embeddings_layers, age_embeddings_layers


def compute_interaction(data, model, vocabulary, args):
    if is_wrapped(model):
        model = model.module
    model = model.to(args.device)
    interaction_partners = load_interaction_partners(args.interaction_partners)
    index_of_interest = (
        load_required_indexes(args.codes_required_in_trajectory, vocabulary)
        if args.codes_required_in_trajectory
        else None
    )
    test_data_loader = get_dataset_loader(args, data)
    attributable_code_layers, _ = get_attributable_layers(model, args.modalities)
    lig_code = LayerIntegratedGradients(model, attributable_code_layers)
    test_iterator = iter(test_data_loader)
    progress_bar = tqdm(total=len(test_iterator))
    code_attribute_collection = {
        "pid": [],
        "removed_code": [],
        "code": [],
        "modality": [],
        "days_to_censor": [],
        "attribution": [],
        "time_to_trajectory_end": [],
    }
    i = 0
    code_parser = CodeParser(args.atc_level, args.diag_level, 5)

    for batch in test_iterator:
        if batch is None:
            i += 1
            continue
        for interaction_partner in [None] + interaction_partners:
            run_batch = deepcopy(batch)
            run_batch = {
                k: v.clone().detach() for k, v in run_batch.items() if isinstance(v, torch.Tensor)
            }
            if interaction_partner:
                t, m = interaction_partner
                if (run_batch[f"{m}_tokens"] == t).sum() == 0:
                    continue
                run_batch[f"{m}_tokens"][run_batch[f"{m}_tokens"] == t] = vocabulary.code_to_index(
                    PAD_CODE, m
                )

            run_batch = train.prepare_batch(run_batch, args, model.inputs)
            model_inputs = get_inputs_for_attributions(run_batch, model)
            (attr, _) = attribute_batch(
                explain_code=lig_code,
                explain_age=None,
                batch=model_inputs,
                internal_batch_size=len(run_batch["patient_id"]) * 3,
                month_idx=len(args.month_endpoints) - 1,
            )
            for (
                modality,
                code_attributes,
            ) in zip(args.modalities, attr):
                if len(code_attributes.shape) == 2:  # happens if only 1 patient in batch
                    code_attributes = np.expand_dims(code_attributes, 0)
                try:
                    for patient_tokens, code_attribute, pid, days, time_seq in zip(
                        run_batch[f"{modality}_tokens"],
                        code_attributes,
                        run_batch["patient_id"],
                        run_batch["days_to_censor"],
                        run_batch["time_seq"],
                    ):
                        attributable_indexes = (
                            patient_tokens != vocabulary.code_to_index(PAD_CODE, modality)
                        ).nonzero()
                        for index in attributable_indexes:
                            if index_of_interest:
                                if (
                                    not patient_tokens[index[0], index[1]].item()
                                    in index_of_interest[modality]
                                ):
                                    continue

                            try:
                                code = vocabulary.index_to_code(
                                    patient_tokens[index[0], index[1]].item(), modality
                                )
                                code, _ = code_parser.parse_code(code, modality)
                                attribution = code_attribute[index[0], index[1]].item()
                                code_attribute_collection["attribution"].append(attribution)
                                code_attribute_collection["pid"].append(pid.item())
                                code_attribute_collection["code"].append(code)
                                code_attribute_collection["removed_code"].append(
                                    vocabulary.index_to_code(
                                        interaction_partner[0], interaction_partner[1]
                                    )
                                    if interaction_partner
                                    else None
                                )
                                code_attribute_collection["modality"].append(modality)
                                code_attribute_collection["days_to_censor"].append(days.item())
                                code_attribute_collection["time_to_trajectory_end"].append(
                                    time_seq[index[0]].item()
                                )

                            except KeyError:
                                print(
                                    f"Skipping index {patient_tokens[index[0], index[1]].item()}, as it was not in vocabulary"
                                )
                                continue
                except TypeError:
                    print("Skipping batch")
                    continue
        i += 1
        progress_bar.update(1)
    return code_attribute_collection


def compute_attribution(attribute_data, model, vocabulary, args):
    if is_wrapped(model):
        model = model.module
    model = model.to(args.device)
    test_data_loader = get_dataset_loader(args, attribute_data)

    attributable_code_layers, attributable_age_layers = get_attributable_layers(
        model, args.modalities
    )

    lig_code = LayerIntegratedGradients(model, attributable_code_layers)
    if attributable_age_layers:
        lig_age = LayerIntegratedGradients(model, attributable_age_layers)
    else:
        lig_age = None

    test_iterator = iter(test_data_loader)
    progress_bar = tqdm(total=len(test_iterator))
    code_attribute_collection = {
        "pid": [],
        "code": [],
        "modality": [],
        "days_to_censor": [],
        "attribution": [],
        "time_to_trajectory_end": [],
    }
    age_attribute_collection = {
        "pid": [],
        "age": [],
        "days_to_censor": [],
        "attribution": [],
    }
    i = 0
    code_parser = CodeParser(args.atc_level, args.diag_level, 5)

    for batch in test_iterator:
        if batch is None:
            print("Empty batch")
            continue
        batch = train.prepare_batch(batch, args, model.inputs)
        ages = (batch["age"] // 365).squeeze().tolist()
        model_inputs = get_inputs_for_attributions(batch, model)
        (
            attr,
            age_attr,
        ) = attribute_batch(
            lig_code,
            lig_age,
            model_inputs,
            internal_batch_size=len(batch["patient_id"]),
            month_idx=len(args.month_endpoints) - 1,
        )

        for (
            modality,
            code_attributes,
        ) in zip(args.modalities, attr):
            try:
                for patient_tokens, code_attribute, pid, age, days, time_seq in zip(
                    batch[f"{modality}_tokens"],
                    code_attributes,
                    batch["patient_id"],
                    ages,
                    batch["days_to_censor"],
                    batch["time_seq"],
                ):
                    attributable_indexes = (
                        patient_tokens != vocabulary.code_to_index(PAD_CODE, modality)
                    ).nonzero()
                    for index in attributable_indexes:
                        try:
                            code = vocabulary.index_to_code(
                                patient_tokens[index[0], index[1]].item(), modality
                            )
                            code, _ = code_parser.parse_code(code, modality)
                            attribution = code_attribute[index[0], index[1]].item()
                            code_attribute_collection["attribution"].append(attribution)
                            code_attribute_collection["pid"].append(pid.item())
                            code_attribute_collection["code"].append(code)
                            code_attribute_collection["modality"].append(modality)
                            code_attribute_collection["days_to_censor"].append(days.item())
                            code_attribute_collection["time_to_trajectory_end"].append(
                                time_seq[index[0]].item()
                            )

                        except KeyError:
                            print(
                                f"Skipping index {patient_tokens[index[0], index[1]].item()}, as it was not in vocabulary"
                            )
                            continue
            except TypeError:
                print("Skipping batch")
                continue
        try:
            for age_attribute, pid, age, days in zip(
                age_attr,
                batch["patient_id"],
                ages,
                batch["days_to_censor"],
            ):
                age_attribute_collection["pid"].append(pid.item())
                age_attribute_collection["days_to_censor"].append(days.item())
                age_attribute_collection["age"].append(age)
                age_attribute_collection["attribution"].append(age_attribute.item())
        except TypeError:
            print("Skipping batch")
            continue

        i += 1
        if i % int(0.1 * len(test_iterator)) == 0:
            progress_bar.update(int(0.1 * len(test_iterator)))

    return code_attribute_collection, age_attribute_collection


def attribute_batch(explain_code, explain_age, batch, internal_batch_size, month_idx):
    batch_age = deepcopy(batch)
    if explain_code:
        attributions_code = explain_code.attribute(
            inputs=batch,
            return_convergence_delta=False,
            target=month_idx,
            internal_batch_size=internal_batch_size,
            n_steps=25,
        )
        code_attributtions = []
        for i in range(len(attributions_code)):
            attributions_code[i] = attributions_code[i].sum(dim=-1).squeeze(0)
            attributions_code[i] = attributions_code[i] / torch.norm(attributions_code[i])
            if len(attributions_code[i].shape) == 1:
                attributions_code[i] = attributions_code[i].unsqueeze(1)
            code_attributtions.append(attributions_code[i].cpu().detach().numpy())

    else:
        code_attributtions = []

    if explain_age:
        attributions_age = explain_age.attribute(
            inputs=batch_age,
            return_convergence_delta=False,
            target=month_idx,
            attribute_to_layer_input=True,
            internal_batch_size=internal_batch_size,
        )
        attributions_age[0] = attributions_age[0].sum(dim=(-1, -2)).squeeze()
        attributions_age[0] = attributions_age[0] / torch.norm(attributions_age[0])
        attributions_age[1] = attributions_age[1].sum(dim=(-1, -2)).squeeze()
        attributions_age[1] = attributions_age[1] / torch.norm(attributions_age[1])

        age_attribution = attributions_age[0] + attributions_age[1]

    else:
        age_attribution = []

    return code_attributtions, age_attribution


def get_inputs_for_attributions(batch, model):
    """
    Get arguments in the position order, so they can be passed to model
    """
    batch_size = batch["time_seq"].shape[0]
    return tuple(
        [
            batch[k].to(model.args.device)
            if k in batch.keys()
            else torch.zeros(size=(batch_size, 1))
            for k in model.inputs
        ]
    )
