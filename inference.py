import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

IMAGE_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.webp', '.gif', '.jxl')
TARGET_SIZE = (256, 256)
model = None

def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]

def get_device():
    import torch
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def chunk_sequence(items, ratios, max_len=200, overlap=30):
    """
    Yields slices of (chunk_items, chunk_ratios, start_idx, end_idx) 
    using a sliding window with a fixed page overlap.
    """
    total = len(items)
    if total <= max_len:
        yield items, ratios, 0, total
        return

    start = 0
    while start < total:
        end = min(start + max_len, total)
        
        # If a tiny remainder chunk is left at the end, slide the window 
        # backward to maintain maximum possible sequence context length
        if end - start < overlap and start > 0:
            start = max(0, end - max_len)

        if end == total and (end - start) < max_len and start > 0:
            start = max(0, end - max_len)

        yield items[start:end], ratios[start:end], start, end

        start += (max_len - overlap)

def load(model_path):
    import torch
    from page_sequence_classifier import PageSequenceClassifier

    device = get_device()
    print(f"Loading {model_path} on {device}...")
    
    global model
    model = PageSequenceClassifier(3, 256).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    clean_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

# --- Core Preprocessing & Inference Architecture ---
def preprocess_single_image(source, file_identifier):
    """
    Decodes and resizes a single image from a folder path, ZipFile, or direct BytesIO stream.
    """
    from PIL import Image
    import zipfile

    if isinstance(file_identifier, BytesIO):
        # Handle direct multipart upload streams
        stream = file_identifier
    elif isinstance(source, zipfile.ZipFile):
        # Handle CBZ / Zip archives
        raw_bytes = source.read(file_identifier)
        stream = BytesIO(raw_bytes)
    else:
        # Handle standard folder directories
        stream = os.path.join(source, file_identifier)

    with Image.open(stream) as img:
        ratio = img.size[0] / img.size[1]
        resized_img = img.convert("L").resize(TARGET_SIZE)
        
    return resized_img, ratio

def pipeline_preprocess_batch(source, file_identifiers, num_workers=4):
    """Parallelizes image opening and downsampling across N threads."""
    processed_images = []
    ratios = []

    # ThreadPoolExecutor speeds up the I/O and PIL decompression overhead
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(preprocess_single_image, source, fid) 
            for fid in file_identifiers
        ]
        for future in futures:
            resized_img, ratio = future.result()
            processed_images.append(resized_img)
            ratios.append(ratio)

    return processed_images, ratios

def run_inference(processed_images, ratios) -> list[int]:
    """
    Unified inference execution engine. Seamlessly chunks long inputs 
    using an overlap window before executing the PyTorch forward pass.
    """
    import torch
    from torchvision import transforms
    from util import extract_edges, merge_pages_rtl
    from PIL import Image

    device = get_device()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    total_pages = len(processed_images)
    preds = [None] * total_pages

    # Extract dynamic limits straight from the loaded architecture class
    MAX_LEN = getattr(model, "MAX_SEQUENCE_LENGTH", 200)
    OVERLAP = 50

    # Execute inference over window segments
    for chunk_imgs, chunk_rats, start, end in chunk_sequence(processed_images, ratios, MAX_LEN, OVERLAP):
        
        ratios_tensor = torch.tensor(chunk_rats, dtype=torch.float32).unsqueeze(0).to(device)
        seq_len = len(chunk_imgs)
        lengths_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)

        left_edge_tensors = []
        right_edge_tensors = []
        for img in chunk_imgs:
            left_edge, right_edge = extract_edges(img)
            left_edge_tensors.append(transform(left_edge))
            right_edge_tensors.append(transform(right_edge))

        left_edges_batch = torch.stack(left_edge_tensors).unsqueeze(0).to(device)
        right_edges_batch = torch.stack(right_edge_tensors).unsqueeze(0).to(device)

        spread_tensors = []
        for i in range(seq_len - 1):
            spread_pil = merge_pages_rtl(chunk_imgs[i], chunk_imgs[i + 1])
            spread_tensors.append(transform(spread_pil))

        spread_tensors.append(transform(Image.new("L", (64, 256))))
        spreads_batch = torch.stack(spread_tensors).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(
                left_edges_batch,
                right_edges_batch,
                spreads_batch,
                ratios_tensor,
                lengths_tensor,
            )
            probs = torch.softmax(logits, dim=-1)[0]
            chunk_preds = torch.argmax(probs, dim=-1).cpu().tolist()

        # Re-aggregate window output chunks into global list positions
        for local_idx, global_idx in enumerate(range(start, end)):
            preds[global_idx] = chunk_preds[local_idx]

    return preds

# --- Web Server App setup ---
def create_app(num_workers):
    import zipfile
    from io import BytesIO
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Page Sequence Classifier API")

    @app.post("/upload")
    async def process_uploaded_files(files: list[UploadFile]):
        """Processes a batch of multipart form-data image uploads in parallel."""
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Read file streams concurrently from the request payload
        file_data_streams = []
        for file in files:
            data = await file.read()
            file_data_streams.append(BytesIO(data))

        # Pass the memory streams into our unified parallel processing engine
        # We pass None as the source since each stream is completely self-contained
        images, ratios = pipeline_preprocess_batch(
            source=None, 
            file_identifiers=file_data_streams, 
            num_workers=num_workers
        )

        preds = run_inference(images, ratios)
        return JSONResponse(content=preds)

    @app.post("/cbz")
    async def process_local_cbz(path: str):
        if not os.path.isfile(path) or not path.lower().endswith('.cbz'):
            raise HTTPException(status_code=400, detail="Invalid path or not a .cbz file.")

        try:
            with zipfile.ZipFile(path, 'r') as z:
                image_files = [f for f in z.namelist() if f.lower().endswith(IMAGE_EXTENSIONS)]
                image_files.sort(key=natural_sort_key)
                
                if not image_files:
                    raise HTTPException(status_code=400, detail="No valid images found in CBZ.")

                images, ratios = pipeline_preprocess_batch(z, image_files, num_workers=num_workers)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="The file provided is a corrupted zip archive.")

        # Run inference 
        preds = run_inference(images, ratios)
        
        # Build an ordered list of structured items
        ordered_results = [
            {"filename": name, "side": pred}
            for name, pred in zip(image_files, preds)
        ]
        
        return JSONResponse(content=ordered_results)
    return app

def handle_local_directory(folder_path: str, num_workers: int):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    file_names = os.listdir(folder_path)
    image_files = [f for f in file_names if f.lower().endswith(IMAGE_EXTENSIONS)]
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print(f"No valid images found in {folder_path}")
        return

    # Process directory using the exact same centralized parallel function
    images, ratios = pipeline_preprocess_batch(folder_path, image_files, num_workers=num_workers)
    
    preds = run_inference(images, ratios)
    print(preds)

if __name__ == "__main__":
    # Get the physical/logical core count of the machine dynamically
    default_cores = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None

    parser = argparse.ArgumentParser(description="Run page sequence classification inference.")
    parser.add_argument("command", choices=["serve", "run"])
    parser.add_argument("--folder", required=False)
    parser.add_argument("--model", default="best_comic_classifier.pth")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument(
        "-t", "--threads", 
        type=int, 
        default=default_cores, 
        help=f"Number of worker threads (default: system cores = {default_cores})"
    )

    args = parser.parse_args()
  
    if args.command == "run" and not args.folder:
        parser.error("--folder is required when command is 'run'")

    load(args.model)

    if args.command == "serve":
        import uvicorn
        app = create_app(num_workers=args.threads)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.command == "run":
        handle_local_directory(args.folder, num_workers=args.threads)
