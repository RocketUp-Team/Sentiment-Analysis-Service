"""Custom HuggingFace Trainer with weighted cross-entropy loss.

``WeightedLossTrainer`` là subclass của ``transformers.Trainer`` — override duy nhất
``compute_loss()`` để áp dụng class weights vào CrossEntropyLoss.

Tất cả các tính năng khác (gradient accumulation, mixed precision, LoRA support,
MLflow callback, evaluation loop, checkpoint) hoạt động giống hệt Trainer gốc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """HuggingFace Trainer với weighted cross-entropy loss.

    Args:
        class_weights: FloatTensor shape ``(num_labels,)`` — weight cho từng class.
                       Thường được tạo bởi ``compute_class_weights()``.
        *args, **kwargs: Forwarded sang ``transformers.Trainer.__init__``.

    Example::

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=collator,
            class_weights=weights_tensor,
        )
        trainer.train()
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Override compute_loss để inject class weights vào CrossEntropyLoss.

        Khi ``return_outputs=False`` (training step): chỉ trả về scalar loss.
        Khi ``return_outputs=True`` (evaluation step): trả về (loss, outputs).
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Di chuyển weights sang cùng device với logits (CPU / CUDA / MPS)
        weights = self._class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
