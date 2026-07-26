import csv
import os
import re

import numpy as np
import torch
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from page_sequence_classifier import PageSequenceClassifier
from util import extract_edges, merge_pages_rtl

DATASET_ROOT = "./dataset_resized"
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True


class IterationSpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


progress_bar = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    IterationSpeedColumn(),
)


def parse_chapter_modulo(filename):
    name_without_ext = os.path.splitext(filename)[0]
    numbers = re.findall(r"\d+", name_without_ext)
    if numbers:
        n = int(numbers[-1])
        return n % 3
    return None


class FlatNPZDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted(
            [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith(".npz")
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with np.load(file_path, mmap_mode="r") as data:
            images_np = np.array(data["images"])
            labels_np = np.array(data["labels"])
            ratios_np = np.array(data["aspect_ratios"])

        seq_len = len(images_np)

        left_edge_imgs = []
        right_edge_imgs = []
        for img_np in images_np:
            left_edge, right_edge = extract_edges(img_np)
            left_edge_imgs.append(left_edge)
            right_edge_imgs.append(right_edge)

        spread_imgs = []
        for i in range(seq_len - 1):
            spread = merge_pages_rtl(images_np[i], images_np[i + 1])
            spread_imgs.append(spread)

        spread_imgs.append(Image.new("L", (64, 256)))

        filename = os.path.basename(file_path)
        return (
            left_edge_imgs,
            right_edge_imgs,
            spread_imgs,
            labels_np,
            ratios_np,
            filename,
        )


class GaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


class MarginJitter:
    def __init__(self, max_shift):
        self.max_shift = max_shift

    def __call__(self, img):
        """img: PIL Image (grayscale, mode 'L')"""
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return img

        img_np = np.asarray(img)
        _h, w = img_np.shape

        if shift > 0:
            border = img_np[:, 0:1]
            pad = np.repeat(border, shift, axis=1)
            shifted = np.hstack([pad, img_np[:, : w - shift]])
        else:
            shift = abs(shift)
            border = img_np[:, -1:]
            pad = np.repeat(border, shift, axis=1)
            shifted = np.hstack([img_np[:, shift:], pad])

        return Image.fromarray(shifted, mode="L")


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None, jitter_ratio=False, margin_jitter=None):
        self.subset = subset
        self.transform = transform
        self.jitter_ratio = jitter_ratio
        self.margin_jitter = margin_jitter

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        left_edge_imgs, right_edge_imgs, spread_imgs, labels, ratios, filename = (
            self.subset[idx]
        )

        if self.margin_jitter:
            left_edge_imgs = [self.margin_jitter(img) for img in left_edge_imgs]
            right_edge_imgs = [self.margin_jitter(img) for img in right_edge_imgs]

        if self.transform:
            left_edge_imgs = [self.transform(img) for img in left_edge_imgs]
            right_edge_imgs = [self.transform(img) for img in right_edge_imgs]
            spread_imgs = [self.transform(img) for img in spread_imgs]

        if self.jitter_ratio:
            noise = np.random.uniform(0.92, 1.08, size=ratios.shape)
            ratios = ratios * noise

        left_edges_tensor = torch.stack(left_edge_imgs)
        right_edges_tensor = torch.stack(right_edge_imgs)
        spreads_tensor = torch.stack(spread_imgs)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        ratios_tensor = torch.tensor(ratios, dtype=torch.float32)

        return (
            left_edges_tensor,
            right_edges_tensor,
            spreads_tensor,
            labels_tensor,
            ratios_tensor,
            filename,
        )


def collate_variable_sequences(batch):
    left_edges = [item[0] for item in batch]
    right_edges = [item[1] for item in batch]
    spreads = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    ratios = [item[4] for item in batch]
    filenames = [item[5] for item in batch]

    lengths = torch.tensor([x.size(0) for x in left_edges], dtype=torch.long)
    padded_left_edges = nn.utils.rnn.pad_sequence(left_edges, True, 0.0)
    padded_right_edges = nn.utils.rnn.pad_sequence(right_edges, True, 0.0)
    padded_spreads = nn.utils.rnn.pad_sequence(spreads, True, 0.0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, True, -100)
    padded_ratios = nn.utils.rnn.pad_sequence(ratios, True, 0.0)

    return (
        padded_left_edges,
        padded_right_edges,
        padded_spreads,
        padded_labels,
        padded_ratios,
        lengths,
        filenames,
    )


def main():
    epoch_pred_dir = "epoch_predictions"
    os.makedirs(epoch_pred_dir, exist_ok=True)

    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    csv_log_path = "training_log.csv"
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_dataset = FlatNPZDataset(root_dir=DATASET_ROOT)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_raw, test_raw = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_dataset_base = TransformedSubset(
        train_raw,
        transform=train_transform,
        jitter_ratio=True,
        margin_jitter=MarginJitter(max_shift=8),
    )
    test_dataset = TransformedSubset(
        test_raw, transform=test_transform, jitter_ratio=False
    )

    EPOCH_MULTIPLIER = 1
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_base] * EPOCH_MULTIPLIER
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_variable_sequences,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_sequences,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = PageSequenceClassifier(3, 256).to(DEVICE)
    model = torch.compile(model)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if ("edge_encoder" in name or "spread_encoder" in name) and param.requires_grad:
            backbone_params.append(param)
        elif param.requires_grad:
            head_params.append(param)

    optimizer = optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": 1e-5,
            },
            {"params": head_params, "lr": LEARNING_RATE},
        ],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    scaler = torch.amp.GradScaler(device=DEVICE.type) if DEVICE.type == "cuda" else None

    best_test_loss = float("inf")

    with progress_bar as progress:
        for epoch in range(NUM_EPOCHS):
            # TRAIN
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            train_task = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", total=len(train_loader)
            )

            for batch_idx, (
                left_edges,
                right_edges,
                spreads,
                targets,
                ratios,
                lengths,
                _,
            ) in enumerate(train_loader):
                left_edges = left_edges.to(DEVICE, non_blocking=True)
                right_edges = right_edges.to(DEVICE, non_blocking=True)
                spreads = spreads.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                ratios = ratios.to(DEVICE, non_blocking=True)
                lengths = lengths.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if DEVICE.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(
                            left_edges, right_edges, spreads, ratios, lengths
                        )
                        logits_flat = logits.view(-1, 3)
                        targets_flat = targets.view(-1)
                        loss = criterion(logits_flat, targets_flat)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(left_edges, right_edges, spreads, ratios, lengths)
                    logits_flat = logits.view(-1, 3)
                    targets_flat = targets.view(-1)
                    loss = criterion(logits_flat, targets_flat)
                    loss.backward()
                    optimizer.step()

                valid_mask = targets_flat != -100
                train_loss += loss.item()
                _, preds = torch.max(logits_flat, dim=1)
                train_correct += torch.sum((preds == targets_flat) & valid_mask).item()
                train_total += torch.sum(valid_mask).item()

                progress.update(train_task, advance=1)

            progress.remove_task(train_task)
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = (train_correct / train_total) * 100

            # TEST
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            epoch_predictions_log = []

            group_correct = {0: 0, 1: 0, 2: 0}
            group_total = {0: 0, 1: 0, 2: 0}

            test_task = progress.add_task(
                f"[green]Epoch {epoch + 1}/{NUM_EPOCHS} [Test]", total=len(test_loader)
            )

            with torch.no_grad():
                for (
                    left_edges,
                    right_edges,
                    spreads,
                    targets,
                    ratios,
                    lengths,
                    filenames,
                ) in test_loader:
                    left_edges = left_edges.to(DEVICE, non_blocking=True)
                    right_edges = right_edges.to(DEVICE, non_blocking=True)
                    spreads = spreads.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE, non_blocking=True)
                    ratios = ratios.to(DEVICE, non_blocking=True)
                    lengths = lengths.to(DEVICE, non_blocking=True)

                    if DEVICE.type == "cuda":
                        with torch.amp.autocast(
                            device_type="cuda", dtype=torch.float16
                        ):
                            logits = model(
                                left_edges, right_edges, spreads, ratios, lengths
                            )
                            logits_flat = logits.view(-1, 3)
                            targets_flat = targets.view(-1)
                            loss = criterion(logits_flat, targets_flat)
                    else:
                        logits = model(
                            left_edges, right_edges, spreads, ratios, lengths
                        )
                        logits_flat = logits.view(-1, 3)
                        targets_flat = targets.view(-1)
                        loss = criterion(logits_flat, targets_flat)

                    preds_shaped = torch.argmax(logits, dim=-1)  # [Batch, Max_Seq]
                    for b in range(targets.size(0)):
                        seq_len = lengths[b].item()
                        file_name = filenames[b]

                        true_array = targets[b, :seq_len].cpu().tolist()
                        pred_array = preds_shaped[b, :seq_len].cpu().tolist()

                        epoch_predictions_log.append(
                            {
                                "chapter": file_name,
                                "targets": true_array,
                                "predictions": pred_array,
                            }
                        )

                        group_idx = parse_chapter_modulo(file_name)
                        if group_idx in [0, 1, 2]:
                            for t_val, p_val in zip(true_array, pred_array):
                                if t_val == p_val:
                                    group_correct[group_idx] += 1
                                group_total[group_idx] += 1

                    valid_mask = targets_flat != -100
                    test_loss += loss.item()
                    _, preds = torch.max(logits_flat, dim=1)
                    test_correct += torch.sum(
                        (preds == targets_flat) & valid_mask
                    ).item()
                    test_total += torch.sum(valid_mask).item()

                    progress.update(test_task, advance=1)

            progress.remove_task(test_task)
            epoch_test_loss = test_loss / len(test_loader)
            epoch_test_acc = (test_correct / test_total) * 100

            group_accs = {}
            for g in [0, 1, 2]:
                if group_total[g] > 0:
                    group_accs[g] = (group_correct[g] / group_total[g]) * 100
                else:
                    group_accs[g] = 0.0

            current_head_lr = optimizer.param_groups[1]["lr"]

            with open(csv_log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        epoch_train_loss,
                        epoch_train_acc,
                        epoch_test_loss,
                        epoch_test_acc,
                    ]
                )

            progress.console.print(f"[bold]Epoch [{epoch + 1}/{NUM_EPOCHS}][/bold]")
            progress.console.print(f"  Learning Rate (Head): {current_head_lr:.6f}")
            progress.console.print(
                f"  Train -> Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%"
            )
            progress.console.print(
                f"  Test  -> Loss: {epoch_test_loss:.4f} | Accuracy: {epoch_test_acc:.2f}%"
            )
            progress.console.print(
                f"    - Group (n % 3 == 0) Accuracy: {group_accs[0]:.2f}% ({group_correct[0]}/{group_total[0]})"
            )
            progress.console.print(
                f"    - Group (n % 3 == 1) Accuracy: {group_accs[1]:.2f}% ({group_correct[1]}/{group_total[1]})"
            )
            progress.console.print(
                f"    - Group (n % 3 == 2) Accuracy: {group_accs[2]:.2f}% ({group_correct[2]}/{group_total[2]})"
            )

            epoch_txt_path = os.path.join(
                epoch_pred_dir, f"test_predictions_epoch_{epoch + 1}.txt"
            )
            with open(epoch_txt_path, "w") as f:
                for item in epoch_predictions_log:
                    f.write(f"[{item['chapter']}]\n")
                    f.write(f"{'Targets:':<13}{item['targets']}\n")
                    f.write(f"{'Predictions:':<13}{item['predictions']}\n")
                    f.write("\n")

            epoch_model_path = os.path.join(
                checkpoints_dir, f"comic_classifier_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), epoch_model_path)

            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                torch.save(model.state_dict(), "best_comic_classifier.pth")
                progress.console.print("  [green] Best model weights updated[/green]")

            print("-" * 50)

            scheduler.step()

    print("\nTraining and evaluation finished.")


if __name__ == "__main__":
    main()
