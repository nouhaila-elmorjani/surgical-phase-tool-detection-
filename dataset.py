"""Deprecated compatibility wrapper for dataset utilities.

The canonical implementations now live in
``surgical_phase_tool.dataset``. This module is kept only so that any
old imports like ``import dataset`` continue to work without shipping
duplicate logic.
"""

from surgical_phase_tool.dataset import *                   


def load_manifest(manifest_path: str) -> List[Dict]:
    samples: List[Dict] = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def build_video_index(samples: List[Dict]) -> Dict[str, List[int]]:
    video_to_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(samples):
        video = row["video"]
        video_to_indices.setdefault(video, []).append(idx)
                                                    
    for video, idxs in video_to_indices.items():
        idxs.sort(key=lambda i: int(samples[i]["frame_number"]))
    return video_to_indices


class MultiTaskWindowDataset(Dataset):
    def __init__(self, manifest_path: str, is_train: bool = True):
        self.samples = load_manifest(manifest_path)
        self.video_index = build_video_index(self.samples)
        self.is_train = is_train

                                          
        transforms_list = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if is_train:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))

        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = T.Compose(transforms_list)

        self.valid_indices = self._compute_valid_center_indices(WINDOW_SIZE)

    def _compute_valid_center_indices(self, window_size: int) -> List[int]:
        half = window_size // 2
        valid: List[int] = []
        for video, idxs in self.video_index.items():
            if len(idxs) < window_size:
                continue
            for i_pos, global_idx in enumerate(idxs):
                if i_pos - half < 0 or i_pos + half >= len(idxs):
                    continue
                valid.append(global_idx)
        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_image(self, rel_path: str) -> Image.Image:
        abs_path = os.path.join(IMAGE_ROOT, os.path.relpath(rel_path, "data/processed")) if rel_path.startswith("data/processed") else os.path.join(IMAGE_ROOT, rel_path)
        return Image.open(abs_path).convert("RGB")

    def _get_window_indices(self, center_idx: int) -> List[int]:
        row = self.samples[center_idx]
        video = row["video"]
        idxs = self.video_index[video]
                                                                                                 
        pos = idxs.index(center_idx)
        half = WINDOW_SIZE // 2
        window_idxs = idxs[pos - half : pos + half + 1]
        assert len(window_idxs) == WINDOW_SIZE
        return window_idxs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center_idx = self.valid_indices[index]
        window_sample_indices = self._get_window_indices(center_idx)

        images = []
        for idx in window_sample_indices:
            row = self.samples[idx]
            img = self._load_image(row["filepath"])
            img = self.transform(img)
            images.append(img)
                             
        frames = torch.stack(images, dim=0)

        center_row = self.samples[center_idx]
        phase_name = center_row["phase"]
        phase_id = PHASE_TO_ID[phase_name]
        phase_target = torch.zeros(NUM_PHASE_CLASSES, dtype=torch.float32)
        phase_target[phase_id] = 1.0

        tool_target = torch.zeros(NUM_TOOL_CLASSES, dtype=torch.float32)
        for i, tool in enumerate(TOOL_COLUMNS):
            val = int(center_row[tool])
            tool_target[i] = float(val)

        return frames, phase_target, tool_target
