import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy
from rich.progress import Progress


def join(f1, f2):
    im1 = cv2.imread(f1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread(f2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    return numpy.hstack([im1, im2])


def sort(f):
    f = os.path.splitext(os.path.basename(f))[0]
    r = re.findall(r"([0-9]+)", f)
    r = [int(s) for s in r]
    if not r:
        if f == "cover":
            return [0]
        print(f)
        sys.exit(1)
    return r


m = 0

folders = os.listdir("src")
for folder in folders:
    src = os.path.join("src", folder)
    out = os.path.join("dataset_root", folder)
    if os.path.exists(out + "-1"):
        continue

    files = os.listdir(src)
    files = [f for f in files if f.endswith((".jpeg", ".jpg"))]
    files = [os.path.join(src, f) for f in files]
    files = sorted(files, key=sort)

    # 0: RIGHT 1: LEFT 2: SPREAD
    files = [(n % 2, f) for n, f in enumerate(files, 1)]

    n = 0
    with Progress() as progress:
        task = progress.add_task(folder, total=len(files))
        while n + 30 < len(files):
            length = random.randint(15, 50)
            chapter = files[n : n + length]
            n += length
            progress.update(task, completed=n)
            m += 1

            if random.random() < 0.33:
                if chapter[0][0] == 1:
                    del chapter[0]
            else:
                if chapter[0][0] == 0:
                    del chapter[0]

            if random.random() > 0.25:
                for i in range(int(random.triangular(0, length // 4, mode=2))):
                    if not [c for c in chapter[:-1] if c[0] == 0]:
                        break

                    c = [c for c in enumerate(chapter[:-1]) if c[1][0] == 0]
                    v = random.choice(c)[0]

                    chapter[v] = (2, chapter[v + 1][1], chapter[v][1])
                    del chapter[v + 1]

                    if v + 1 < len(chapter) and random.random() > 0.7:
                        del chapter[v + 1]

            dirname = f"{out}-{m}"
            os.makedirs(dirname, exist_ok=True)

            for i in range(random.randint(0, 3)):
                if not [c for c in chapter if c[0] == 0]:
                    break

                c = [c for c in enumerate(chapter) if c[1][0] == 0]
                del chapter[random.choice(c)[0]]

            def fun(f, out_path):
                if f[0] == 2:
                    im = join(f[1], f[2])
                else:
                    im = cv2.imread(f[1])
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                numpy.savez_compressed(out_path, im)

            with ThreadPoolExecutor(max_workers=16) as executor:
                for i, f in enumerate(chapter):
                    out_path = os.path.join(dirname, f"{i}.npz")
                    executor.submit(fun, f, out_path)

            with open(os.path.join(dirname, "labels.json"), "w+") as jf:
                json.dump([f[0] for f in chapter], jf)
