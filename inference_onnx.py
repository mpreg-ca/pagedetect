import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from PIL import Image

from util import extract_edges, merge_pages_rtl

TARGET_SIZE = (256, 256)
MAX_SEQUENCE_LENGTH = 200
OVERLAP = 50

ort.preload_dlls()


def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


def chunk_sequence(items, ratios, max_len, overlap):
    total = len(items)
    if total <= max_len:
        yield items, ratios, 0, total
        return

    start = 0
    while start < total:
        end = min(start + max_len, total)

        if end == total and (end - start) < max_len and start > 0:
            start = max(0, end - max_len)

        yield items[start:end], ratios[start:end], start, end

        start += max_len - overlap
        print(start, end, total)
        if end >= total - 1:
            break


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def preprocess_single_image(img: Image):
    ratio = img.size[0] / img.size[1]
    resized_img = img.convert("L").resize(TARGET_SIZE)

    return resized_img, ratio


def pipeline_preprocess_batch(images: list[Image]):
    processed_images = []
    ratios = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(preprocess_single_image, image) for image in images]
        for future in futures:
            resized_img, ratio = future.result()
            processed_images.append(resized_img)
            ratios.append(ratio)

    return processed_images, ratios


def normalize(img_array: np.ndarray) -> np.ndarray:
    return (img_array.astype(np.float32) / 255.0 - 0.5) / 0.5


class Session:
    def __init__(self):
        self.session = None

    def load(self, model_path):
        print(f"Loading ONNX model: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = []
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            print("Using CUDA execution provider")
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
            print("Using CoreML execution provider")

        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            model_path, sess_options, providers=providers
        )

    def run_inference(self, images) -> list[int]:
        images, ratios = pipeline_preprocess_batch(images)

        total_pages = len(images)
        preds = [None] * total_pages

        for chunk_imgs, chunk_rats, start, end in chunk_sequence(
            images, ratios, MAX_SEQUENCE_LENGTH, OVERLAP
        ):
            seq_len = len(chunk_imgs)

            # Extract edge features
            left_edge_arrays = []
            right_edge_arrays = []
            for img in chunk_imgs:
                left_edge, right_edge = extract_edges(img)
                left_arr = normalize(np.array(left_edge))[np.newaxis, :, :]  # [1, H, W]
                right_arr = normalize(np.array(right_edge))[
                    np.newaxis, :, :
                ]  # [1, H, W]
                left_edge_arrays.append(left_arr)
                right_edge_arrays.append(right_arr)

            # Stack edges: [Seq, 1, H, W] -> [1, Seq, 1, H, W]
            left_edges = np.stack(left_edge_arrays)[np.newaxis, :, :, :, :]
            right_edges = np.stack(right_edge_arrays)[np.newaxis, :, :, :, :]

            # Create spread features
            spread_arrays = []
            for i in range(seq_len - 1):
                spread_pil = merge_pages_rtl(chunk_imgs[i], chunk_imgs[i + 1])
                spread_arr = normalize(np.array(spread_pil))[np.newaxis, :, :]
                spread_arrays.append(spread_arr)

            # Add padding for last image (no next page to stitch)
            padding = np.zeros((1, 256, 64), dtype=np.float32)
            spread_arrays.append(padding)

            # Stack spreads: [Seq, 1, H, W] -> [1, Seq, 1, H, W]
            spreads = np.stack(spread_arrays)[np.newaxis, :, :, :, :]

            # Prepare ratios: [1, Seq]
            ratios_array = np.array(chunk_rats, dtype=np.float32).reshape(1, -1)

            # Pad to model's fixed sequence length if needed (for non-dynamic models)
            actual_seq_len = seq_len

            # Run ONNX inference
            inputs = {
                "left_edges": left_edges.astype(np.float32),
                "right_edges": right_edges.astype(np.float32),
                "spreads": spreads.astype(np.float32),
                "ratios": ratios_array,
            }

            outputs = self.session.run(None, inputs)
            logits = outputs[0]  # [1, Seq, 3]

            # Get predictions (only for actual sequence, not padding)
            probs = softmax(logits, axis=-1)[0][:actual_seq_len]  # [Seq, 3]
            chunk_preds = np.argmax(probs, axis=-1).tolist()  # [Seq]
            print(chunk_preds)

            for local_idx, global_idx in enumerate(range(start, end)):
                preds[global_idx] = chunk_preds[local_idx]

        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run page sequence classification inference."
    )
    parser.add_argument("folder")
    parser.add_argument("--model", default="page_classifier.onnx")

    args = parser.parse_args()

    session = Session()
    session.load(args.model)

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".gif", ".jxl")

    files = os.listdir(args.folder)
    files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort(key=natural_sort_key)
    files = [os.path.join(args.folder, f) for f in files]

    if not files:
        print(f"No valid images found in {args.folder}")
        sys.exit(1)

    images = []
    for f in files:
        images.append(Image.open(f))

    preds = session.run_inference(images)
    print(preds)
