# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dataloaders and dataset utils."""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_requirements,
    check_yaml,
    clean_str,
    cv2,
    is_colab,
    is_kaggle,
    segments2boxes,
    unzip_file,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = "See https://docs.ultralytics.com/nexus/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose().

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Inherit from DistributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    """A distributed sampler ensuring deterministic shuffling and balanced data distribution across GPUs."""

    def __iter__(self):
        """Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
):
    """Creates and returns a configured DataLoader instance for loading and processing image datasets."""
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        drop_last=quad,
        pin_memory=PIN_MEMORY,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a repeating sampler that cycles through the given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Yields indices from the sampler indefinitely by resetting when exhausted."""
        while True:
            yield from iter(self.sampler)


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        rank=-1,
    ):
        """Initializes the dataset loader with image paths, labels, and various processing options."""
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        # Advanced augmentation flags
        self.mosaic9 = self.augment and hyp.get("mosaic9", False)  # Mosaic9 augmentation
        self.mixup = self.augment and hyp.get("mixup", 0.0) > 0  # MixUp augmentation
        self.copy_paste = self.augment and hyp.get("copy_paste", 0.0) > 0  # Copy-Paste augmentation
        self.randaugment = self.augment and hyp.get("randaugment", False)  # RandAugment policy

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[rect]  # wh
            self.segments = [self.segments[i] for i in irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, nb = 0, np.ceil(n / batch_size).astype(int)  # number of batches
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(n))
                pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    if cache_images == "disk":
                        self.ims[i], self.im_hw0[i], self.im_hw[i] = None, None, None  # .npy saved
                    else:
                        self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # images, hw_original, hw_resized
                    pbar.desc = f"{prefix}Caching images ({b + 1}/{nb})"
                pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Checks if available RAM is sufficient for caching images, considering a safety margin."""
        b, gb = 0, 1 << 30  # bytes, GiB
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f'{"caching images ✅" if cache else "not caching images ⚠️"}'
            )
        return cache

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Cache dataset labels, check images and read shapes, verifying image-label correspondence."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files),
                bar_format=TQDM_BAR_FORMAT,
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING: No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(str(path), x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.im_files)

    def load_image(self, i):
        """Loads an image by index, returning the image, its original dimensions, and resized dimensions."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for training, combining images and labels with random perspective augmentation."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(labels4, segments4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        """Loads a 9-image mosaic for training, combining images and labels with random perspective augmentation."""
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 9 tiles
                h0, w0 = min(h, s), min(w, s)  # center tile size
                x1a, y1a, x2a, y2a = s, s, s + w0, s + h0  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = 0, 0, w0, h0  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top
                x1a, y1a, x2a, y2a = s, max(s - h, 0), s + w, s
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), w, h
            elif i == 2:  # top right
                x1a, y1a, x2a, y2a = s + w, max(s - h, 0), min(s + 2 * w, 3 * s), s
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 3:  # right
                x1a, y1a, x2a, y2a = s + w, s, min(s + 2 * w, 3 * s), s + h
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), h
            elif i == 4:  # bottom right
                x1a, y1a, x2a, y2a = s + w, s + h, min(s + 2 * w, 3 * s), min(s + 2 * h, 3 * s)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            elif i == 5:  # bottom
                x1a, y1a, x2a, y2a = s, s + h, s + w, min(s + 2 * h, 3 * s)
                x1b, y1b, x2b, y2b = 0, 0, w, min(h, y2a - y1a)
            elif i == 6:  # bottom left
                x1a, y1a, x2a, y2a = max(s - w, 0), s + h, s, min(s + 2 * h, 3 * s)
                x1b, y1b, x2b, y2b = w - min(w, x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 7:  # left
                x1a, y1a, x2a, y2a = max(s - w, 0), s, s, s + h
                x1b, y1b, x2b, y2b = w - min(w, x2a - x1a), 0, w, h
            elif i == 8:  # top left
                x1a, y1a, x2a, y2a = max(s - w, 0), max(s - h, 0), s, s
                x1b, y1b, x2b, y2b = w - min(w, x2a - x1a), h - min(h, y2a - y1a), w, h

            img9[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img9[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 3 * s, out=x)  # clip when using random_perspective()

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9

    def __getitem__(self, index):
        """Returns a single image and labels after applying augmentations, suitable for model training."""
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        mosaic9 = self.mosaic9 and random.random() < hyp.get("mosaic9_prob", 0.5)
        if mosaic9:
            img, labels = self.load_mosaic9(index)
        elif mosaic:
            img, labels = self.load_mosaic(index)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic and not mosaic9:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

            # MixUp augmentation
            if self.mixup and random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

            # Copy-Paste augmentation
            if self.copy_paste and random.random() < hyp["copy_paste"]:
                img, labels = copy_paste(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

            # RandAugment
            if self.randaugment:
                img = self.apply_randaugment(img)

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations
            if self.albumentations is not None:
                img, labels = self.albumentations(img, labels)

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def apply_randaugment(self, img):
        """Applies RandAugment to the image using torchvision's implementation."""
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Apply RandAugment
        transform = torchvision.transforms.RandAugment(
            num_ops=self.hyp.get("randaugment_num_ops", 2),
            magnitude=self.hyp.get("randaugment_magnitude", 9)
        )
        img_pil = transform(img_pil)
        
        # Convert back to numpy array (BGR)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches, formatting images and labels for model input."""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """Collates data into batches of 4 images, creating mosaics for training."""
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                    0
                ].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat(
                    (
                        torch.cat((img[i], img[i + 1]), 1),
                        torch.cat((img[i + 2], img[i + 3]), 1),
                    ),
                    2,
                )
                l = torch.cat(
                    (
                        label[i],
                        label[i + 1] + ho,
                        label[i + 2] + wo,
                        label[i + 3] + ho + wo,
                    ),
                    0,
                ) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def img2label_paths(img_paths):
    """Defines label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class HubDatasetStats:
    """Class for generating Hub dataset statistics, supporting both classification and object detection tasks."""

    def __init__(self, path="coco128.yaml", autodownload=False):
        """Initializes dataset statistics generator for classification and detection tasks."""
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors="ignore") as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data["path"] = data_dir
        except Exception as e:
            raise Exception("error/HUB/dataset_stats/yaml_load") from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data["path"] + "-hub")
        self.im_dir = self.hub_dir / "images"
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {"nc": data["nc"], "names": data["names"]}  # statistics dictionary
        self.data = data

    @staticmethod
    def _unzip(path):
        """Unzips a dataset zip file to a specified directory, verifying its integrity."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, path.parent, path
        else:
            unzip_file(path, path.parent)
            return True, str(path.stem), str(path.with_suffix(".yaml"))

    def _hub_ops(self, f, max_dim=1920):
        """Resizes and saves images for HUB, supporting both classification and detection formats."""
        f_new = self.im_dir / f.name  # dataset-agnostic location
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, "JPEG", quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            LOGGER.info(f"WARNING: HUB ops PIL failure {f}: {e}")
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im.width)  # ratio
            if r < 1:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        """Generates dataset JSON for Ultralytics HUB, including image stats and class information."""
        def _round(labels):
            """Updates labels to integer class and 4 decimal place floats."""
            if self.data["task"] == "detect":
                return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]
            elif self.data["task"] == "segment":
                return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]
            else:
                return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

        for split in "train", "val", "test":
            self.stats[split] = None  # predefine
            path = self.data.get(split)

            # Check dataset
            if path is None:  # not split
                continue
            files = [p for p in Path(path).rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS]  # image files
            if not files:
                continue
            if self.data["task"] == "classify":
                # Classify dataset (folder structure: /class_name/image.jpg)
                from torchvision.datasets import ImageFolder

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes), dtype=int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            elif self.data["task"] in ("detect", "segment"):
                # Detection/Segmentation dataset (labels in /labels/*.txt)
                from utils.dataloaders import LoadImagesAndLabels

                dataset = LoadImagesAndLabels(path)  # load dataset
                x = np.array(
                    [
                        np.bincount(label[:, 0].astype(int), minlength=self.data["nc"])
                        for label in dataset.labels
                        if len(label)
                    ]
                )  # shape(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.sum(x.sum(1) == 0)),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [
                        {Path(f).name: _round(l.tolist())} for f, l in zip(dataset.im_files, dataset.labels) if len(l)
                    ],
                }

        # Save, print and return
        if save:
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """Compresses a single image to reduced size while preserving aspect ratio."""
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f"WARNING: PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(f_new or f), im)


def verify_image_label(args):
    """Verifies a single image-label pair, ensuring integrity and correctness of dataset files."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"WARNING: {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and len(lb[0]) == 6:  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"WARNING: {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path="coco128.yaml", autodownload=False, verbose=False):
    """Return dataset statistics and optionally print them."""
    stats = HubDatasetStats(path, autodownload)
    stats.get_json(verbose)
    return stats.stats


def model_scale(depth_multiple, width_multiple, max_channels=1024):
    """
    Scale a YOLOv5 model based on depth and width multipliers.
    
    Args:
        depth_multiple: Depth scaling factor (number of layers)
        width_multiple: Width scaling factor (number of channels)
        max_channels: Maximum number of channels
    
    Returns:
        Scaled model configuration
    """
    # Base configuration for YOLOv5s
    base_config = {
        "depth_multiple": 0.33,  # model depth multiple
        "width_multiple": 0.50,  # layer channel multiple
        "anchors": [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326],  # P5/32
        ],
        "backbone": [
            [-1, 1, "Conv", [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
            [-1, 3, "C3", [128]],
            [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
            [-1, 6, "C3", [256]],
            [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
            [-1, 9, "C3", [512]],
            [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
            [-1, 3, "C3", [1024]],
            [-1, 1, "SPPF", [1024, 5]],  # 9
        ],
        "head": [
            [-1, 1, "Conv", [512, 1, 1]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
            [-1, 3, "C3", [512, False]],  # 13
            [-1, 1, "Conv", [256, 1, 1]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
            [-1, 3, "C3", [256, False]],  # 17 (P3/8-small)
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat head P4
            [-1, 3, "C3", [512, False]],  # 20 (P4/16-medium)
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],  # cat head P5
            [-1, 3, "C3", [1024, False]],  # 23 (P5/32-large)
            [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],  # Detect(P3, P4, P5)
        ],
    }
    
    # Scale depth (number of layers)
    scaled_config = base_config.copy()
    
    # Scale backbone
    for i, layer in enumerate(scaled_config["backbone"]):
        if layer[2] == "C3":
            # Scale number of repeats
            repeats = max(round(layer[3] * depth_multiple), 1) if len(layer) > 3 else 1
            scaled_config["backbone"][i] = layer[:3] + [repeats] + layer[4:] if len(layer) > 3 else layer[:3] + [repeats]
    
    # Scale head
    for i, layer in enumerate(scaled_config["head"]):
        if layer[2] == "C3":
            # Scale number of repeats
            repeats = max(round(layer[3] * depth_multiple), 1) if len(layer) > 3 else 1
            scaled_config["head"][i] = layer[:3] + [repeats] + layer[4:] if len(layer) > 3 else layer[:3] + [repeats]
    
    # Scale width (number of channels)
    def scale_channels(channels):
        """Scale channels based on width_multiple."""
        return min(max(round(channels * width_multiple), 1), max_channels)
    
    # Scale backbone channels
    for i, layer in enumerate(scaled_config["backbone"]):
        if layer[2] == "Conv":
            # Scale output channels
            out_channels = scale_channels(layer[3][0])
            kernel_size = layer[3][1] if len(layer[3]) > 1 else 3
            stride = layer[3][2] if len(layer[3]) > 2 else 1
            padding = layer[3][3] if len(layer[3]) > 3 else None
            scaled_config["backbone"][i] = layer[:3] + [[out_channels, kernel_size, stride, padding]] if padding else layer[:3] + [[out_channels, kernel_size, stride]]
        elif layer[2] == "C3":
            # Scale C3 channels
            out_channels = scale_channels(layer[3][0]) if len(layer) > 3 and isinstance(layer[3], list) else scale_channels(layer[3])
            scaled_config["backbone"][i] = layer[:3] + [out_channels] + layer[4:] if len(layer) > 3 else layer[:3] + [out_channels]
        elif layer[2] == "SPPF":
            # Scale SPPF channels
            out_channels = scale_channels(layer[3][0])
            kernel_size = layer[3][1] if len(layer[3]) > 1 else 5
            scaled_config["backbone"][i] = layer[:3] + [[out_channels, kernel_size]]
    
    # Scale head channels
    for i, layer in enumerate(scaled_config["head"]):
        if layer[2] == "Conv":
            # Scale output channels
            out_channels = scale_channels(layer[3][0])
            kernel_size = layer[3][1] if len(layer[3]) > 1 else 1
            stride = layer[3][2] if len(layer[3]) > 2 else 1
            padding = layer[3][3] if len(layer[3]) > 3 else None
            scaled_config["head"][i] = layer[:3] + [[out_channels, kernel_size, stride, padding]] if padding else layer[:3] + [[out_channels, kernel_size, stride]]
        elif layer[2] == "C3":
            # Scale C3 channels
            out_channels = scale_channels(layer[3][0]) if len(layer) > 3 and isinstance(layer[3], list) else scale_channels(layer[3])
            scaled_config["head"][i] = layer[:3] + [out_channels] + layer[4:] if len(layer) > 3 else layer[:3] + [out_channels]
    
    # Update scaling factors
    scaled_config["depth_multiple"] = depth_multiple
    scaled_config["width_multiple"] = width_multiple
    
    return scaled_config


def generate_model_variants():
    """Generate different YOLOv5 model variants using compound scaling."""
    variants = {
        "nexusn": {"depth_multiple": 0.33, "width_multiple": 0.25, "description": "Nano - smallest, fastest"},
        "nexuss": {"depth_multiple": 0.33, "width_multiple": 0.50, "description": "Small - good balance"},
        "nexusm": {"depth_multiple": 0.67, "width_multiple": 0.75, "description": "Medium - better accuracy"},
        "nexusl": {"depth_multiple": 1.0, "width_multiple": 1.0, "description": "Large - high accuracy"},
        "nexusx": {"depth_multiple": 1.33, "width_multiple": 1.25, "description": "XLarge - highest accuracy"},
        "nexusn6": {"depth_multiple": 0.33, "width_multiple": 0.25, "description": "Nano P6 - for 1280 images"},
        "nexuss6": {"depth_multiple": 0.33, "width_multiple": 0.50, "description": "Small P6 - for 1280 images"},
        "nexusm6": {"depth_multiple": 0.67, "width_multiple": 0.75, "description": "Medium P6 - for 1280 images"},
        "nexusl6": {"depth_multiple": 1.0, "width_multiple": 1.0, "description": "Large P6 - for 1280 images"},
        "nexusx6": {"depth_multiple": 1.33, "width_multiple": 1.25, "description": "XL P6 - for 1280 images"},
    }
    
    configs = {}
    for name, params in variants.items():
        config = model_scale(params["depth_multiple"], params["width_multiple"])
        config["description"] = params["description"]
        configs[name] = config
    
    return configs


def auto_scale_model(target_latency_ms=None, target_accuracy=None, base_model="nexuss"):
    """
    Automatically scale a YOLOv5 model based on target latency or accuracy.
    
    Args:
        target_latency_ms: Target inference latency in milliseconds
        target_accuracy: Target mAP accuracy (0-1)
        base_model: Base model to scale from
    
    Returns:
        Scaled model configuration
    """
    # Base model configurations
    base_configs = {
        "nexusn": {"depth_multiple": 0.33, "width_multiple": 0.25, "latency_ms": 1.0, "accuracy": 0.28},
        "nexuss": {"depth_multiple": 0.33, "width_multiple": 0.50, "latency_ms": 2.0, "accuracy": 0.37},
        "nexusm": {"depth_multiple": 0.67, "width_multiple": 0.75, "latency_ms": 4.0, "accuracy": 0.45},
        "nexusl": {"depth_multiple": 1.0, "width_multiple": 1.0, "latency_ms": 8.0, "accuracy": 0.49},
        "nexusx": {"depth_multiple": 1.33, "width_multiple": 1.25, "latency_ms": 16.0, "accuracy": 0.51},
    }
    
    if base_model not in base_configs:
        base_model = "nexuss"
    
    config = base_configs[base_model].copy()
    
    if target_latency_ms is not None:
        # Scale based on latency target (simplified model)
        # Latency scales roughly with depth * width^2
        current_latency = config["latency_ms"]
        scale_factor = (target_latency_ms / current_latency) ** 0.5
        
        # Adjust depth and width
        config["depth_multiple"] *= scale_factor ** 0.5
        config["width_multiple"] *= scale_factor ** 0.5
        
        # Clamp to reasonable values
        config["depth_multiple"] = max(0.2, min(config["depth_multiple"], 1.5))
        config["width_multiple"] = max(0.2, min(config["width_multiple"], 1.5))
        
        # Update estimated latency
        config["latency_ms"] = target_latency_ms
    
    elif target_accuracy is not None:
        # Scale based on accuracy target (simplified model)
        # Accuracy scales roughly with log(depth) and log(width)
        current_accuracy = config["accuracy"]
        
        # Simple linear approximation
        accuracy_diff = target_accuracy - current_accuracy
        
        # Adjust depth and width
        config["depth_multiple"] *= (1 + accuracy_diff * 2)
        config["width_multiple"] *= (1 + accuracy_diff * 2)
        
        # Clamp to reasonable values
        config["depth_multiple"] = max(0.2, min(config["depth_multiple"], 1.5))
        config["width_multiple"] = max(0.2, min(config["width_multiple"], 1.5))
        
        # Update estimated accuracy
        config["accuracy"] = target_accuracy
    
    # Generate scaled model configuration
    scaled_config = model_scale(config["depth_multiple"], config["width_multiple"])
    scaled_config["estimated_latency_ms"] = config.get("latency_ms")
    scaled_config["estimated_accuracy"] = config.get("accuracy")
    
    return scaled_config