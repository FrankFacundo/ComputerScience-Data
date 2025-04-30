import os
from pathlib import Path
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb  # <-- Import W&B
from dataset_cppe5 import CPPE5Dataset  # Assuming this file exists locally
from datasets import load_from_disk
from image_transform import (
    train_augmentation_and_transform,
    validation_transform,
)  # Assuming these exist locally
from mape_evaluator import MAPEvaluator  # Assuming this exists locally
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoImageProcessor,
    RTDetrForObjectDetection,
    Trainer,
    TrainingArguments,
)

# --- W&B Configuration ---
# Set your W&B project and entity (username or team name)
# You can also set these as environment variables: WANDB_PROJECT, WANDB_ENTITY
os.environ["WANDB_PROJECT"] = "rtdetr-layout"
os.environ["WANDB_ENTITY"] = "frankfacundo"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log model checkpoints as W&B Artifacts
os.environ["WANDB_LOG_MODEL"] = (
    "false"  # Disable saving model checkpoints as W&B Artifacts
)


# Ensure you are logged into W&B. Run `wandb login` in your terminal if needed.
# -------------------------


def main(mode):
    if mode == "train":
        train()
    elif mode == "inference":
        inference()
    else:
        print(f"Unknown mode: {mode}. Choose 'train' or 'inference'.")


def download_models(
    local_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    if not progress:
        disable_progress_bars()
    download_path = snapshot_download(
        repo_id="ds4sd/docling-models",
        force_download=force,
        local_dir=local_dir,
        revision="v2.1.0",
    )
    return Path(download_path)


def train():
    # --- Configuration ---
    # checkpoint = "PekingU/rtdetr_r50vd_coco_o365"
    _model_path = "model_artifacts/layout"
    artifacts_path = download_models(".", progress=True) / _model_path
    image_size = 480
    output_dir = "rtdetr-exams-finetune-wandb_2"  # Changed output dir slightly
    os.makedirs(output_dir, exist_ok=True)  # Create if it doesn't exist
    # run_name = f"rtdetr-finetune-exams-{image_size}"  # W&B Run Name

    # --- Dataset Loading ---
    print("Loading dataset...")
    dataset = get_dataset()
    print("Filtering dataset...")
    dataset = dataset.filter(all_bboxes_valid)
    print(f"Dataset sizes: { {k: len(v) for k, v in dataset.items()} }")

    # --- Labels ---
    id2label = {
        0: "background",
        1: "caption",
        2: "footnote",
        3: "formula",
        4: "list_item",
        5: "page_footer",
        6: "page_header",
        7: "picture",
        8: "section_header",
        9: "table",
        10: "text",
        11: "title",
        12: "document_index",
        13: "code",
        14: "checkbox_selected",
        15: "checkbox_unselected",
        16: "form",
        17: "key_value_region",
    }
    # id2label = {
    #     1: "background",
    #     2: "caption",
    #     3: "footnote",
    #     4: "formula",
    #     5: "list_item",
    #     6: "page_footer",
    #     7: "page_header",
    #     8: "picture",
    #     9: "section_header",
    #     10: "table",
    #     11: "text",
    #     12: "title",
    #     13: "document_index",
    #     14: "code",
    #     15: "checkbox_selected",
    #     16: "checkbox_unselected",
    #     17: "form",
    #     18: "key_value_region",
    # }
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    # --- Image Processor ---
    print("Loading image processor...")
    image_processor = AutoImageProcessor.from_pretrained(
        artifacts_path,
        do_resize=True,
        size={"width": image_size, "height": image_size},
        use_fast=True,
    )

    # --- Datasets Preparation ---
    print("Preparing datasets...")
    train_dataset = CPPE5Dataset(
        dataset["train"], image_processor, transform=train_augmentation_and_transform
    )
    validation_dataset = CPPE5Dataset(
        dataset["validation"], image_processor, transform=validation_transform
    )
    test_dataset = CPPE5Dataset(
        dataset["test"], image_processor, transform=validation_transform
    )
    print(
        f"Prepared dataset sizes: train={len(train_dataset)}, val={len(validation_dataset)}, test={len(test_dataset)}"
    )

    # --- Evaluator ---
    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=image_processor, threshold=0.01, id2label=id2label
    )

    # --- Model Loading ---
    print("Loading model...")
    # Option 1: Finetune from COCO pre-trained (comment out if using Option 2)
    # model = AutoModelForObjectDetection.from_pretrained(
    #     checkpoint,
    #     id2label=id2label,
    #     label2id=label2id,
    #     num_labels=num_labels, # Explicitly set num_labels
    #     ignore_mismatched_sizes=True, # Allow head size mismatch
    # )

    # Option 2: Load specific pre-trained model (like DocLing)

    try:
        model_config_path = os.path.join(str(artifacts_path), "config.json")
        model = RTDetrForObjectDetection.from_pretrained(
            str(artifacts_path), config=model_config_path
        )
        # Potentially adjust the classification head if number of labels differs
        # Example: Check if config needs updating (this might be complex depending on model specifics)
        if model.config.num_labels != num_labels:
            print(
                f"Warning: Model has {model.config.num_labels} labels, dataset has {num_labels}. Adjusting head..."
            )
            # Re-initialize the classification head - specific implementation depends on RTDetr architecture
            # This is a placeholder - you might need a more specific way to resize the head
            model.class_labels_classifier = torch.nn.Linear(
                model.config.d_model, num_labels + 1
            )  # +1 for background? check RTDetr
            model.config.id2label = id2label
            model.config.label2id = label2id
            model.config.num_labels = num_labels  # Update config

    except Exception as e:
        print(f"Could not load model from {artifacts_path}: {e}")
        print("Falling back to COCO pre-trained model.")
        model = RTDetrForObjectDetection.from_pretrained(
            artifacts_path,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model.to(device)
    print(f"Model loaded on {device}.")

    # --- Training Arguments ---
    # print("Defining training arguments...")
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     # --- Core Training Parameters ---
    #     num_train_epochs=50,  # Keep low for testing, increase for real training
    #     learning_rate=5e-5,
    #     max_grad_norm=0.1,
    #     warmup_steps=300,
    #     per_device_train_batch_size=12,  # Adjusted batch size, change based on GPU memory
    #     per_device_eval_batch_size=12,  # Adjusted batch size
    #     gradient_accumulation_steps=1,
    #     # --- Optimizer & Scheduler ---
    #     optim="adamw_torch",
    #     # --- Performance & Technical ---
    #     dataloader_num_workers=4,
    #     # --- W&B Integration ---
    #     report_to="wandb",  # <-- Enable W&B logging
    #     run_name=run_name,  # <-- Set the name for the W&B run
    #     # -----------------------
    #     metric_for_best_model="eval_map",  # Ensure this matches a key in compute_metrics output
    #     greater_is_better=True,
    #     # --- Evaluation & Saving ---
    #     eval_strategy="epoch",  # Changed from 'steps' to 'epoch' for consistency
    #     save_strategy="epoch",  # Changed from 'steps' to 'epoch'
    #     # evaluation_strategy="epoch", # Redundant if eval_strategy is set
    #     # save_steps=500, # Used if strategy is 'steps'
    #     # eval_steps=500, # Used if strategy is 'steps'
    #     save_total_limit=2,
    #     load_best_model_at_end=True,
    #     remove_unused_columns=False,
    #     eval_do_concat_batches=False,  # Important for object detection evaluation
    #     # --- Logging & Reporting ---
    #     logging_strategy="steps",  # How often to log loss to console/W&B
    #     logging_steps=50,  # Log every 50 steps
    #     # fp16=torch.cuda.is_available(), # Enable mixed precision if using GPU
    # )

    # --- Recommended Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,  # Your output directory
        run_name=f"rtdetr-finetune-exams-bs{12 * 1}-lr{5e-5}-ep{50}",  # Example descriptive run name
        # --- Core Training Parameters ---
        num_train_epochs=15,  # Increased significantly, adjust based on convergence
        learning_rate=5e-5,  # Common starting point, experiment with 1e-5, 2e-5, 1e-4
        per_device_train_batch_size=12,  # START HERE, Adjust based on VRAM. Try 4, 8, 12, 16...
        # per_device_eval_batch_size=12,  # Usually same as train batch size
        gradient_accumulation_steps=1,  # Increase if per_device_train_batch_size is low due to memory
        # (e.g., batch_size=4, accum_steps=4 => effective_bs=16)
        # --- Optimizer & Scheduler ---
        optim="adamw_torch",  # Standard optimizer
        # lr_scheduler_type="cosine",  # Often works well
        # warmup_ratio=0.1,  # Warmup over 10% of total training steps
        # weight_decay=0.01,  # Regularization to prevent overfitting
        max_grad_norm=0.1,  # Gradient clipping (often recommended for DETR)
        # --- Evaluation & Saving ---
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        save_total_limit=3,  # Keep only the best + last 2 checkpoints
        load_best_model_at_end=True,  # Reload the best model found during training
        metric_for_best_model="eval_map",  # Your key metric from MAPEvaluator
        greater_is_better=True,  # mAP should be maximized
        # --- Logging & Reporting ---
        logging_strategy="steps",  # Log metrics every N steps
        logging_steps=50,  # Log training loss etc. every 50 steps
        report_to="wandb",  # Log to Weights & Biases
        # --- Performance & Technical ---
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,  # Use bf16 if available (preferred over fp16 on compatible hardware)
        dataloader_num_workers=4,  # Adjust based on your CPU/IO, 4-8 is common
        remove_unused_columns=False,  # Keep as False for custom datasets/collators
        eval_do_concat_batches=False,  # Crucial for object detection metrics
    )

    # --- Trainer Initialization ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=image_processor,  # Pass the processor, it's used like a tokenizer here
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # --- Training ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        trainer.save_model()  # Save the final best model
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Ensure W&B run is finished even if training fails
        wandb.finish()
        raise e

    # --- Evaluation ---
    print("Starting final evaluation on test set...")
    try:
        metrics = trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="test"
        )  # Use "test" prefix
        trainer.log_metrics("test", metrics)  # Changed from "eval" to "test"
        trainer.save_metrics("test", metrics)  # Changed from "eval" to "test"
        print("Test set evaluation metrics:")
        pprint(metrics)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        # Ensure W&B run is finished even if evaluation fails
        wandb.finish()
        raise e

    # --- Finish W&B Run ---
    wandb.finish()
    print("W&B run finished.")


def inference():
    # Inference code remains largely the same
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_image_path = "test2.png"  # Make sure this image exists
    if not os.path.exists(local_image_path):
        print(f"Error: Test image '{local_image_path}' not found.")
        return

    image = Image.open(local_image_path).convert("RGB")

    # Use the output directory from training (or specify a checkpoint path)
    # This assumes you trained and saved the model in the `train` function's output_dir
    model_repo = "model_artifacts/layout"  # Original model

    # model_repo = (
    #     "rtdetr-exams-finetune-wandb/checkpoint-800"  # Default to final saved model
    # )
    model_config = os.path.join(str(model_repo), "config.json")
    # Or point to a specific checkpoint:
    # model_repo = "./rtdetr-exams-finetune-wandb/checkpoint-XXXX" # Replace XXXX

    if not os.path.exists(model_repo):
        print(
            f"Error: Model directory '{model_repo}' not found. Train the model first or specify correct path."
        )
        return

    print(f"Loading model and processor from {model_repo} for inference...")
    image_processor = AutoImageProcessor.from_pretrained(model_repo)
    model = RTDetrForObjectDetection.from_pretrained(
        model_repo, config=model_config
    ).to(device)
    print("Model and processor loaded.")

    ## Detect bounding boxes
    print("Performing inference...")
    inputs = image_processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move all inputs to device

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processing
    target_sizes = torch.tensor([image.size[::-1]]).to(
        device
    )  # Ensure target_sizes is on the same device

    results = image_processor.post_process_object_detection(
        outputs, threshold=0.3, target_sizes=target_sizes
    )
    result = results[0]  # Get the first result

    print("\n--- Detection Results ---")
    for score, label, box in zip(
        result["scores"].cpu(), result["labels"].cpu(), result["boxes"].cpu()
    ):
        box_list = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item() + 1]} with confidence "
            f"{round(score.item(), 3)} at location {box_list}"
        )

    ## Plot the result
    print("Drawing bounding boxes on image...")
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)  # Try loading a specific font
    except IOError:
        font = ImageFont.load_default()

    for score, label, box in zip(
        result["scores"].cpu(), result["labels"].cpu(), result["boxes"].cpu()
    ):
        box_list = [round(i, 2) for i in box.tolist()]
        x_min, y_min, x_max, y_max = tuple(box_list)
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
        label_text = model.config.id2label[label.item() + 1]
        text_position = (
            x_min,
            y_min - 15 if y_min > 15 else y_min + 1,
        )  # Adjust text position
        # Simple background for text
        text_bbox = draw.textbbox(text_position, label_text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text(text_position, label_text, fill="white", font=font)

    output_image_path = "test_inference_output.png"
    image_with_boxes.save(output_image_path)
    print(f"Output image saved to {output_image_path}")
    # image_with_boxes.show() # Optionally show image directly


def get_dataset():
    # dataset = load_dataset("cppe-5") # Example
    try:
        dataset = load_from_disk("dataset/dataset_exams_2")
        print("Loaded dataset from disk.")
    except FileNotFoundError:
        print(
            "Error: Dataset not found at 'dataset/dataset_exams'. Please ensure the dataset exists."
        )
        # Optionally download or provide instructions here
        # Example: download_dataset_command()
        raise

    # Ensure validation split exists
    if "validation" not in dataset or "test" not in dataset:
        print("Creating train/validation/test splits...")
        # First split: 80% train, 20% temp
        train_temp_split = dataset["train"].train_test_split(test_size=0.20, seed=42)
        # Second split: temp -> 75% validation, 25% test (relative to temp, so 15% val, 5% test overall)
        val_test_split = train_temp_split["test"].train_test_split(
            test_size=0.25, seed=42
        )  # 0.20 * 0.25 = 0.05

        dataset["train"] = train_temp_split["train"]
        dataset["validation"] = val_test_split["train"]  # 75% of 20% = 15%
        dataset["test"] = val_test_split["test"]  # 25% of 20% = 5%
        print("Splits created.")

    return dataset


# --- Helper Functions (Unchanged unless noted) ---


def all_bboxes_valid(example):
    img_width, img_height = example["image"].size
    if not example["objects"]["bbox"]:  # Handle cases with no bounding boxes
        return True  # Keep images with no objects
    for bbox in example["objects"]["bbox"]:
        if len(bbox) != 4:
            return False  # Invalid format
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        if not (
            0 <= x_min < img_width
            and 0 <= x_max <= img_width
            and 0 <= y_min < img_height
            and 0 <= y_max <= img_height
            and width > 0
            and height > 0
        ):  # Also check width/height > 0
            return False
    return True


# Visualization functions (show_sample, show_transformed_image*, etc.) remain the same
# ... (keep your existing show_* functions here) ...
def show_sample(dataset, id2label):
    # Load image and annotations
    image = dataset["train"][65]["image"]
    annotations = dataset["train"][65]["objects"]

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        class_idx = annotations["category"][i]
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), id2label[class_idx], fill="white")

    image.show()


def show_transformed_image(dataset, id2label):
    for i in [65]:
        image = dataset["train"][i]["image"]
        annotations = dataset["train"][i]["objects"]

        # Apply the augmentation
        output = train_augmentation_and_transform(
            image=np.array(image),
            bboxes=annotations["bbox"],
            category=annotations["category"],
        )

        # Unpack the output
        image = Image.fromarray(output["image"])
        categories, boxes = output["category"], output["bboxes"]

        # Draw the augmented image
        draw = ImageDraw.Draw(image)
        for category, box in zip(categories, boxes):
            print(box)
            x, y, w, h = box
            draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
            draw.text((x, y), id2label[category], fill="white")
        image.show()


def show_transformed_image_2(train_dataset, id2label):
    for i in [65]:
        sample = train_dataset[i]

        # De-normalize image
        image = sample["pixel_values"]
        print("Image tensor shape:", image.shape)
        image = image.numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = Image.fromarray(image.astype(np.uint8))

        # Convert boxes from [center_x, center_y, width, height] to [x, y, width, height] for visualization
        boxes = sample["labels"]["boxes"].numpy()
        print("Boxes shape:", boxes.shape)
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        w, h = image.size
        boxes = boxes * np.array([w, h, w, h])[None]

        categories = sample["labels"]["class_labels"].numpy()
        print("Categories shape:", categories.shape)

        # Draw boxes and labels on image
        draw = ImageDraw.Draw(image)
        for box, category in zip(boxes, categories):
            print(box)
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            draw.text((x, y), id2label[category], fill="white")
        image.show()


def show_transformed_image_3(processed_dataset, id2label, index=0):
    if index >= len(processed_dataset):
        print(
            f"Error: Index {index} is out of bounds for the dataset (size: {len(processed_dataset)})."
        )
        return
    if not id2label:
        print("Error: id2label mapping is required.")
        return

    print(f"--- Visualizing Processed Sample at Index {index} ---")
    sample = processed_dataset[index]
    pixel_values = sample["pixel_values"]
    labels = sample["labels"]
    image_tensor = pixel_values.cpu()
    img_np = image_tensor.numpy()
    min_val, max_val = np.min(img_np), np.max(img_np)
    img_np = (
        ((img_np - min_val) / (max_val - min_val) * 255.0)
        if max_val > min_val
        else np.zeros_like(img_np) + 128
    )
    img_np = img_np.transpose(1, 2, 0)
    image = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    vis_width, vis_height = image.size
    print(f"Visualizing processed image size: ({vis_width}, {vis_height})")
    boxes_normalized = labels["boxes"].cpu().numpy()
    class_labels = labels["class_labels"].cpu().numpy()
    print(f"Found {len(boxes_normalized)} bounding boxes.")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except IOError:
        font = ImageFont.load_default()
    for box_norm, class_id in zip(boxes_normalized, class_labels):
        center_x, center_y, width_norm, height_norm = box_norm
        box_pixel = [
            (center_x - width_norm / 2) * vis_width,
            (center_y - height_norm / 2) * vis_height,
            (center_x + width_norm / 2) * vis_width,
            (center_y + height_norm / 2) * vis_height,
        ]
        label_name = id2label.get(class_id, f"ID:{class_id}")
        draw.rectangle(box_pixel, outline="lime", width=2)
        text_position = (box_pixel[0] + 2, box_pixel[1] + 2)
        text_bbox = draw.textbbox(text_position, label_name, font=font)
        draw.rectangle(text_bbox, fill="lime")
        draw.text(text_position, label_name, fill="black", font=font)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Processed Sample - Index {index}")
    plt.axis("off")
    plt.show()
    print(f"--- Finished Visualizing Sample {index} ---")


def collate_fn(batch):
    # Ensure labels are dictionaries with tensors
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]  # Keep as list of dicts
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    # Simple argument parsing (can be replaced with argparse if needed)
    import sys

    mode = "train"  # Default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()  # Get mode from command line argument

    print(f"Running in mode: {mode}")
    main(mode=mode)
