"""Multimodal vision-language training and evaluation."""

from neuro_agent.multimodal.dataset import (
    load_multimodal_examples,
    normalize_eval_example,
    split_multimodal_by_subjects,
)
from neuro_agent.multimodal.eval import run_multimodal_evaluation
from neuro_agent.multimodal.model import load_vlm_for_inference, load_vlm_for_training, print_architecture_summary
from neuro_agent.multimodal.rlvr_trainer import MultimodalRLVRTrainer, MultimodalRLVRTrainingSummary
from neuro_agent.multimodal.trainer import MultimodalSFTTrainer

__all__ = [
    "MultimodalRLVRTrainer",
    "MultimodalRLVRTrainingSummary",
    "MultimodalSFTTrainer",
    "load_multimodal_examples",
    "load_vlm_for_inference",
    "load_vlm_for_training",
    "normalize_eval_example",
    "print_architecture_summary",
    "run_multimodal_evaluation",
    "split_multimodal_by_subjects",
]
