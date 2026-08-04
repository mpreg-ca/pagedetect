import argparse
import os
import re
import time
import zipfile
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from inference_onnx import Session

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".gif", ".jxl")

app = FastAPI()
session = Session()


def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


@app.post("/")
async def process_uploaded_files(files: list[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    print("Processing")
    t0 = time.time()

    images = []
    for file in files:
        data = await file.read()
        images.append(Image.open(BytesIO(data)))

    preds = session.run_inference(images)

    print("Finished processing in", time.time() - t0)
    return JSONResponse(content=preds)


@app.post("/cbz")
async def process_local_cbz(path: str):
    if not os.path.isfile(path) or not path.lower().endswith(".cbz"):
        raise HTTPException(status_code=400, detail="Invalid path or not a .cbz file.")

    images = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            image_files = [
                f for f in z.namelist() if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
            image_files.sort(key=natural_sort_key)

            if not image_files:
                raise HTTPException(
                    status_code=400, detail="No valid images found in CBZ."
                )

            for f in image_files:
                data = z.read(f)
                images.append(Image.open(BytesIO(data)))

    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400, detail="The file provided is a corrupted zip archive."
        )

    preds = session.run_inference(images)

    ordered_results = [
        {"filename": name, "side": pred} for name, pred in zip(image_files, preds)
    ]

    return JSONResponse(content=ordered_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run page sequence classification inference."
    )
    parser.add_argument("--model", default="page_classifier.onnx")
    parser.add_argument("--port", type=int, default=4000)

    args = parser.parse_args()

    session.load(args.model)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
