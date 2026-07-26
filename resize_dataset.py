import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
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

SRC_DIR = "./dataset_root"
DST_DIR = "./dataset_resized"
TARGET_SIZE = (256, 256)
MAX_WORKERS = 8


class IterationSpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


def to_uint8_gray(img_arr):
    arr = np.array(img_arr, dtype=np.float32)

    if (
        len(arr.shape) == 3
        and arr.shape[0] in [1, 3, 4]
        and arr.shape[2] not in [1, 3, 4]
    ):
        arr = np.transpose(arr, (1, 2, 0))

    min_val, max_val = arr.min(), arr.max()

    if max_val <= 1.0 and min_val >= 0.0:
        arr = arr * 255.0
    elif min_val < 0.0 or max_val > 255.0:
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255.0
        else:
            arr = np.zeros_like(arr)

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if len(arr.shape) == 3:
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.shape[-1] == 3:
            arr = (
                0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            ).astype(np.uint8)
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]

    return arr


def compile_single_sequence(folder, src_dir, dst_dir, target_size):
    src_folder_path = os.path.join(src_dir, folder)
    dst_npz_path = os.path.join(dst_dir, f"{folder}.npz")

    if os.path.exists(dst_npz_path):
        return folder, "SKIPPED"

    npz_files = sorted(
        [f for f in os.listdir(src_folder_path) if f.endswith(".npz")],
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )

    json_files = [f for f in os.listdir(src_folder_path) if f.endswith(".json")]
    if not json_files or not npz_files:
        return folder, "FAILED: Missing .npz or .json files"

    json_path = os.path.join(src_folder_path, json_files[0])
    with open(json_path, "r") as f:
        raw_labels = json.load(f)

    seq_len = min(len(npz_files), len(raw_labels))
    npz_files = npz_files[:seq_len]
    raw_labels = raw_labels[:seq_len]

    resized_images = []
    aspect_ratios = []

    for i, npz_file in enumerate(npz_files):
        npz_path = os.path.join(src_folder_path, npz_file)
        with np.load(npz_path) as data:
            img_key = data.files[0]
            img_arr = np.array(data[img_key])

        gray_arr = to_uint8_gray(img_arr)

        h, w = gray_arr.shape[0], gray_arr.shape[1]
        aspect_ratio = float(w / h)
        aspect_ratios.append(aspect_ratio)

        img_pil = Image.fromarray(gray_arr, mode="L")
        img_resized = img_pil.resize(target_size, Image.Resampling.BILINEAR)
        resized_images.append(np.array(img_resized))

    if len(resized_images) == seq_len:
        np.savez(
            dst_npz_path,
            images=np.array(resized_images),
            labels=np.array(raw_labels),
            aspect_ratios=np.array(aspect_ratios),
        )
        return folder, "SUCCESS"
    else:
        return folder, "FAILED: Skipped due to page mismatch"


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    folders = [
        f
        for f in sorted(os.listdir(SRC_DIR))
        if os.path.isdir(os.path.join(SRC_DIR, f))
    ]

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        IterationSpeedColumn(),
    )

    with progress_bar as progress:
        task = progress.add_task("[cyan]Compiling sequences", total=len(folders))

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    compile_single_sequence, folder, SRC_DIR, DST_DIR, TARGET_SIZE
                ): folder
                for folder in folders
            }

            for future in as_completed(futures):
                folder, status = future.result()

                if status == "SKIPPED":
                    progress.console.print(f"[yellow]Skipped:[/yellow] {folder}")
                elif status.startswith("FAILED"):
                    progress.console.print(f"[red]Failed:[/red] {folder} -> {status}")
                else:
                    progress.console.print(f"[green]Compiled:[/green] {folder}")

                progress.update(task, advance=1)


if __name__ == "__main__":
    main()
