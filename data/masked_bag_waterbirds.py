# import os
# import json
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# from scipy.ndimage import distance_transform_edt
# from pathlib import Path
# from torchvision import transforms
# import torchvision.transforms.functional as TF
# from torch.utils.data import Dataset
# from data.datasets import SpuriousDataset  # AFR base class

# class MaskedBagWaterbirds(SpuriousDataset):
#     def __init__(
#         self,
#         basedir,
#         split="train",
#         transform=None,
#         masking_strategy="mean_fill_dilation",
#         max_distance=30.0,
#         alpha=0.1,
#         crop_scale=(0.85, 1.0),
#         bag_size=5,
#         include_original=True,
#         gradual_shrink=False,
#         current_epoch=0,
#         total_epochs=20,
#     ):
#         super().__init__(basedir, split=split, transform=transform)
#         self.split = split
#         self.root = Path(basedir)
#         self.include_original = include_original
#         self.masking_strategy = masking_strategy
#         self.max_distance = max_distance
#         self.alpha = alpha
#         self.crop_scale = crop_scale
#         self.gradual_shrink = gradual_shrink
#         self.current_epoch = current_epoch
#         self.total_epochs = total_epochs
#         self.bag_size = bag_size

#         # Load segment metadata
#         # Load everything from a single cached JSON file
#         cache_file = os.path.join(self.root, f"waterbirds_split_cache_{split}.json")
#         print(f"📖 Loading Waterbirds cache: {cache_file}")
#         with open(cache_file, "r") as f:
#             cache = json.load(f)

#         self.segments = cache["segments"]
#         self.mask_cache = {}  # batch file -> np.array
#         self.cached_mask_files = cache["mask_files"]  # Optional, if needed

#         # Compose transform (used only if self.transform is None)
#         self.image_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=self.crop_scale),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.25, 0.25, 0.25),
#             transforms.ToTensor()
#         ])

#     def _load_segments_json(self, json_path):
#         with open(json_path, "r") as f:
#             all_data = json.load(f)
#         segments = all_data["segments"]
#         assert len(segments) == len(self.filename_array), "Mismatch between segments and dataset length"
#         return segments

#     def __len__(self):
#         return len(self.filename_array)

#     def __getitem__(self, idx):
#         # Load image
#         file_name = self.filename_array[idx]
#         img_path = os.path.join(self.basedir, file_name)
#         image = Image.open(img_path).convert("RGB")

#         label = self.y_array[idx]
#         group = self.group_array[idx]
#         spurious = self.spurious_array[idx]

#         segment_list = self.segments[idx]

#         # Ensure exactly N segments
#         segment_list = self._pad_or_trim_segments(segment_list, self.bag_size)

#         # Load masks, concepts, clip scores
#         masks, concepts, clip_scores = [], [], []
#         for seg in segment_list:
#             batch_file = seg["batch_file"]
#             mask_index = seg["mask_index"]

#             if batch_file not in self.mask_cache:
#                 self.mask_cache[batch_file] = np.load(self.root / batch_file)

#             mask_array = self.mask_cache[batch_file]
#             mask = mask_array[mask_index]
#             masks.append(torch.tensor(mask))
#             concepts.append(torch.tensor(seg["concepts"], dtype=torch.float32))
#             clip_scores.append(torch.tensor(seg["scores"], dtype=torch.float32))


#         # Apply image + mask transforms
#         original_tensor, masked_bag, _ = self._transform(image, masks)

#         if self.include_original:
#             masked_bag.insert(0, original_tensor)

#         return {
#             "bag": torch.stack(masked_bag),  # [P, C, H, W]
#             "label": label,
#             "clip_score": torch.stack(clip_scores),  # [P,]
#             "concepts": torch.stack(concepts),       # [P, num_concepts]
#             "group": group,
#             "spurious": spurious,
#             "filename": file_name,
#         }

#     def _pad_or_trim_segments(self, segs, target_len):
#         segs = list(segs)
#         if len(segs) >= target_len:
#             return segs[:target_len]
#         else:
#             # repeat entries to fill bag
#             repeat_count = target_len - len(segs)
#             return segs + [segs[i % len(segs)] for i in range(repeat_count)]

#     def _transform(self, img, masks):
#         i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.crop_scale, ratio=(1.0, 1.0))
#         img = TF.resized_crop(img, i, j, h, w, (224, 224))
#         if np.random.rand() < 0.5:
#             img = TF.hflip(img)
#             masks = [TF.hflip(TF.resized_crop(mask.unsqueeze(0), i, j, h, w, (224, 224))).squeeze(0) for mask in masks]
#         else:
#             masks = [TF.resized_crop(mask.unsqueeze(0), i, j, h, w, (224, 224)).squeeze(0) for mask in masks]

#         img = transforms.ColorJitter(0.25, 0.25, 0.25)(img)
#         img_tensor = TF.to_tensor(img)

#         raw_instances = [(mask > 0.5).float() * img_tensor for mask in masks]
#         masked_instances = [self._apply_masking_strategy(img_tensor, mask) for mask in masks]
#         return img_tensor, masked_instances, raw_instances

#     def _apply_masking_strategy(self, img_tensor, mask_tensor):
#         if self.masking_strategy != "mean_fill_dilation":
#             raise ValueError("Only 'mean_fill_dilation' masking is supported in this class.")

#         mask_np = mask_tensor.numpy()
#         mean_val = img_tensor.mean(dim=(1, 2), keepdim=True)

#         if self.gradual_shrink:
#             progress = min(self.current_epoch / self.total_epochs, 1.0)
#             max_dist = self.max_distance - (self.max_distance - 5.0) * progress
#         else:
#             max_dist = self.max_distance

#         dist = distance_transform_edt(1 - mask_np)
#         soft_mask_np = np.ones_like(dist)
#         soft_mask_np[dist > max_dist] = 0.0
#         soft_mask = torch.from_numpy(soft_mask_np).float()

#         return soft_mask * img_tensor + (1 - soft_mask) * mean_val
