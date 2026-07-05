import numpy as np
import cv2
import random as rd
from PIL import Image, ImageOps
import time
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch

SIZE = None  # di-set dari main

def set_seed(seed=13):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def salt_pepper_np(arr, prob=0.05):
    mat = np.random.rand(*arr.shape)
    arr = arr.copy()
    arr[mat < prob / 2] = 0.0
    arr[mat > 1 - prob / 2] = 1.0
    return arr

def concat_img(pil_orig, label):
    img_orig = np.array(pil_orig, dtype=np.float32)
    h, w = img_orig.shape
    flip = np.array(ImageOps.flip(pil_orig), dtype=np.float32)
    mirror = np.array(ImageOps.mirror(pil_orig), dtype=np.float32)
    result = []

    stack_fn = np.hstack if h > w else np.vstack

    for a, b in [(img_orig, mirror), (img_orig, flip), (mirror, flip)]:
        combined = np.array(
            Image.fromarray(stack_fn([a, b]).astype(np.uint8)).resize((SIZE, SIZE)),
            dtype=np.float32
        ) / 255.0
        result.append({"image": combined, "label": label})

    return result

def zoom(pil_orig, label):
    img_orig = np.array(pil_orig, dtype=np.float32)
    h, w = img_orig.shape
    zoom_factor = np.random.uniform(1.15, 1.3)

    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)

    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = img_orig[y1:y1+new_h, x1:x1+new_w]
    zoomed = np.array(Image.fromarray(cropped).resize((w, h)).resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    return zoomed

def augment_single(pil_img, label):
    """Augment satu gambar, return list of {image, label} dicts."""
    result = []

    img_resize = np.array(pil_img.resize((SIZE, SIZE)), dtype=np.float32) / 255.0

    num_rotate = 3
    num_mask = 0

    # Rotasi
    for _ in range(num_rotate):
        rot = np.array(
            pil_img.rotate(np.random.randint(180), expand=True).resize((SIZE, SIZE)),
            dtype=np.float32
        ) / 255.0
        result.append({"image": rot, "label": label})

    # Masking
    for _ in range(num_mask):
        arr = np.array(pil_img, dtype=np.float32)
        h, w = arr.shape
        for _ in range(np.random.randint(3, 7)):
            hs = np.random.randint(0, max(1, int(h - h / 6)))
            ws = np.random.randint(0, max(1, int(w - w / 6)))
            ratio = np.random.randint(4, 7)
            arr[hs:int(hs + h / ratio), ws:int(ws + w / ratio)] = 0
        msk = np.array(Image.fromarray(arr).resize((SIZE, SIZE)), dtype=np.float32) / 255.0
        result.append({"image": msk, "label": label})

    # Original + mirror + flip + zoom
    mirror = np.array(
        ImageOps.mirror(Image.fromarray((img_resize * 255).astype(np.uint8))),
        dtype=np.float32
    ) / 255.0
    flip = np.array(
        ImageOps.flip(Image.fromarray((img_resize * 255).astype(np.uint8))),
        dtype=np.float32
    ) / 255.0
    zoomed = zoom(pil_img, label)

    result.append({"image": img_resize, "label": label})
    result.append({"image": mirror, "label": label})
    result.append({"image": flip, "label": label})
    result.append({"image": zoomed, "label": label})

    return result

def augment_data(data_train, label_counts):
    """Augmentasi sequential tanpa multiprocessing."""
    print(f"\nAugmenting {len(data_train)} images sequentially...")
    t0 = time.time()

    train_img = []
    for i, item in enumerate(data_train):
        try:
            pil_img = item['image'].convert("L")
        except Exception:
            continue
        label = item['label']
        augmented = augment_single(pil_img, label)
        train_img.extend(augmented)
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(data_train)} images, {len(train_img)} augmented...", end='\r')

    print(f"\nAugmentasi selesai: {len(train_img)} gambar  (waktu: {time.time() - t0:.1f}s)")
    return train_img

# ── Dataset classes ──────────────────────────────────────────────────────────────

class PlanktonDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = torch.tensor(item['image'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
        return img, label

class ValTestDataset(Dataset):
    def __init__(self, hf_dataset, size):
        self.dataset = hf_dataset
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['image'].convert("L").resize((self.size, self.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return (torch.tensor(arr, dtype=torch.float32).unsqueeze(0),
                torch.tensor(data['label'], dtype=torch.long))

# ── Data loading & filtering ─────────────────────────────────────────────────────

def load_and_filter_data(ds, maks_sampel=100):
    """Load WHOI plankton dataset, filter label, return train/val/test lists."""
    print("Counting & filtering....")
    label_counts = Counter(ds['train']['label'])
    data_train = []
    data_test = []
    data_val = []

    label_names = ds['train'].features['label'].names
    filtered_labels = {}
    new_map_label = {}
    new_idx = 0
    prohibited_label = []

    for i in label_counts.keys():
        if i not in prohibited_label:
            filtered_labels[i] = 0
            new_map_label[label_names[i]] = (i, new_idx)
            new_idx += 1
    for i in new_map_label.keys():
        print(f"Nama:{i}|Label Lama:{new_map_label[i][0]}|Label Baru:{new_map_label[i][1]}")

    for i in range(len(ds['train'])):
        try:
            ds['train'][i]['image'].convert("L")
        except Exception:
            continue
        data = ds['train'][i]
        if data['label'] in filtered_labels.keys() and filtered_labels[data['label']] < maks_sampel:
            tmp = data.copy()
            tmp['label'] = new_map_label[label_names[data['label']]][1]
            data_train.append(tmp)
            filtered_labels[data['label']] += 1
    print(f"Len Data Train:{len(data_train)}")

    filtered_labels_test = filtered_labels.copy()
    for i in filtered_labels_test.keys():
        filtered_labels_test[i] = 0

    for i in range(len(ds['test'])):
        try:
            ds['test'][i]['image'].convert("L")
        except Exception:
            continue
        data = ds['test'][i]
        if data['label'] in filtered_labels.keys() and filtered_labels_test[data['label']] < maks_sampel:
            tmp = data.copy()
            tmp['label'] = new_map_label[label_names[data['label']]][1]
            data_test.append(tmp)
            filtered_labels_test[data['label']] += 1
    print(f"Len Data Test:{len(data_test)}")

    filtered_labels_val = filtered_labels.copy()
    for i in filtered_labels_val.keys():
        filtered_labels_val[i] = 0

    for i in range(len(ds['validation'])):
        try:
            ds['validation'][i]['image'].convert("L")
        except Exception:
            continue
        data = ds['validation'][i]
        if data['label'] in filtered_labels.keys() and filtered_labels_val[data['label']] < maks_sampel:
            tmp = data.copy()
            tmp['label'] = new_map_label[label_names[data['label']]][1]
            data_val.append(tmp)
            filtered_labels_val[data['label']] += 1
    print(f"Len Data Val:{len(data_val)}")

    return data_train, data_test, data_val, label_counts, new_map_label

def create_dataloaders(train_img, data_val, data_test, batch_size=16):
    """Buat DataLoader untuk train/val/test."""
    train_dataset = PlanktonDataset(train_img)
    val_dataset = ValTestDataset(data_val, SIZE)
    test_dataset = ValTestDataset(data_test, SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

    print(f"Train samples      : {len(train_dataset)}")
    print(f"Train batches/epoch: {len(train_loader)}")
    return train_loader, val_loader, test_loader
