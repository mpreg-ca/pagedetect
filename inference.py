import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from page_sequence_classifier import PageSequenceClassifier
from util import extract_edges, merge_pages_rtl

TARGET_SIZE = (256, 256)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


model = None


def load(model_path):
    print("Loading", model_path)
    global model
    model = PageSequenceClassifier(3, 256).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    clean_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def run_inference(images) -> list[int]:
    with ThreadPoolExecutor(16) as ex:

        def load(im):
            im.load()

        ex.map(load, images)

    ratios = [img.size[0] / img.size[1] for img in images]
    ratios_tensor = torch.tensor(ratios, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    images = [img.convert("L").resize(TARGET_SIZE) for img in images]
    seq_len = len(images)

    lengths_tensor = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    left_edge_tensors = []
    right_edge_tensors = []
    for img in images:
        left_edge, right_edge = extract_edges(img)
        left_edge_tensors.append(transform(left_edge))
        right_edge_tensors.append(transform(right_edge))

    left_edges_batch = torch.stack(left_edge_tensors).unsqueeze(0).to(DEVICE)
    right_edges_batch = torch.stack(right_edge_tensors).unsqueeze(0).to(DEVICE)

    spread_tensors = []
    for i in range(seq_len - 1):
        spread_pil = merge_pages_rtl(images[i], images[i + 1])
        spread_tensors.append(transform(spread_pil))

    spread_tensors.append(transform(Image.new("L", (64, 256))))
    spreads_batch = torch.stack(spread_tensors).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(
            left_edges_batch,
            right_edges_batch,
            spreads_batch,
            ratios_tensor,
            lengths_tensor,
        )  # [1, Seq_Len, 3]
        probs = torch.softmax(logits, dim=-1)[0]  # [Seq_Len, 3]
        preds = torch.argmax(probs, dim=-1).cpu().tolist()  # [Seq_Len]

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve", "run"])
    parser.add_argument("--folder", required=False)
    parser.add_argument("--model", default="best_comic_classifier.pth")
    parser.add_argument("--port", default=4000)

    args = parser.parse_args()
    if args.command == "serve":
        load(args.model)
        app = FastAPI()

        @app.post("/")
        async def post(files: list[UploadFile]):
            images: list[Image.Image] = []
            for file in files:
                data = await file.read()
                images.append(Image.open(BytesIO(data)))

            t0 = time.time()
            preds = run_inference(images)
            print(time.time() - t0)
            print(preds)

            return JSONResponse(preds)

        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        if not os.path.isdir(args.folder):
            print(args.folder, "is not a folder")
            sys.exit(1)

        load(args.model)

        extensions = (".jpeg", ".jpg", ".png", ".webp", ".gif", ".jxl")

        files: list[str] = os.listdir(args.folder)
        files = [f for f in files if f.lower().endswith(extensions)]
        files = [os.path.join(args.folder, f) for f in files]

        images = [Image.open(f) for f in files]

        t0 = time.time()
        preds = run_inference(images)
        print(time.time() - t0)

        print(preds)
