import os
import json
import math
import numpy as np
from PIL import Image
from omegaconf import DictConfig, ListConfig
from typing import Any, Tuple, Optional, List, Dict

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from einops import repeat

class MultiViewDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_views: int,
        bg_color: str,
        img_wh: Tuple[int, int],
        caption_path: str,
        k_near_views: Optional[int] = None,
        sample_views_mode: str = "random",
        start_id: int = 1,
        views_per_layer: Optional[int] = 16,
        add_global_k: Optional[int] = None,
        global_view_ids: Optional[list] = [2, 22, 42, 62],
        num_samples: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
        self.caption_path = caption_path
        self.sample_views_mode = sample_views_mode
        self.start_id = start_id
        self.views_per_layer = views_per_layer
        self.add_global_k = add_global_k
        self.global_view_ids = global_view_ids
        self.k_near_views = k_near_views if k_near_views else num_views
        self.obj_paths = []

        self.load_obj_caption_pairs(caption_path)

        if num_samples is not None:
            self.obj_paths = self.obj_paths[:num_samples]
            self.captions = self.captions[:num_samples]

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_obj_caption_pairs(self, caption_path):
        self.captions = []
        with open(caption_path, "r") as f:
            for line in f:
                obj_id, caption = line.strip().split("\t")
                obj_path = os.path.join(self.root_dir, obj_id)
                # if os.path.exists(obj_path):
                self.obj_paths.append(obj_path)
                self.captions.append(caption)

    def load_image(self, img_path, bg_color, rescale=True, return_type="np"):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if rescale:
            img = img * 2.0 - 1.0  # to -1 ~ 1

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def sample_views(self, sample_mode, num_views_all, num_views):
        if sample_mode == "random":
            view_ids = np.random.choice(num_views_all, num_views, replace=False)
            return view_ids

        assert self.views_per_layer is not None
        if sample_mode.startswith("lay"):
            if len(sample_mode) == 4:
                # sample views from a layer
                start_id = int(sample_mode[3])
                view_ids = np.arange(
                    start_id, num_views_all, num_views_all // num_views
                )[:num_views]
            elif len(sample_mode) == 5:
                # sample views from multiple layers
                layer_start = int(sample_mode[3])
                layer_end = int(sample_mode[4])
                op_id_list = []
                for i in range(layer_start, layer_end + 1):
                    op_id_list += list(
                        range(i, 80, num_views_all // self.views_per_layer)
                    )[: self.views_per_layer]
                op_ids = np.array(op_id_list, dtype=int)
                view_ids = np.random.choice(op_ids, num_views, replace=False)
            else:
                # len(sample_mode) in [4, 5] is required
                raise NotImplementedError
        elif sample_mode == "fixed_random":
            all_ids = np.array(range(0, num_views_all), dtype=np.int32)
            all_ids = all_ids.reshape(self.views_per_layer, -1)

            op_id_list = []
            i = self.start_id
            for line in all_ids:
                op_id_list.append(line[i % 6])
                i += 1
            view_ids = np.array(op_id_list, dtype=int)
        else:
            raise NotImplementedError

        return view_ids

    def get_k_near_views(
        self,
        elevations,
        azimuths,
        k_near_views,
        num_views,
        add_global_k=None,
    ):
        k_near_views = k_near_views + add_global_k if add_global_k else k_near_views
        views = torch.cat((elevations.unsqueeze(1), azimuths.unsqueeze(1)), dim=1)
        distances = torch.cdist(views, views)
        torch.fill_(distances.diagonal(), 0.0)
        k_nearest_indices = torch.topk(distances, k_near_views, largest=False).indices

        if add_global_k is not None:
            global_part = (
                torch.arange(num_views - add_global_k, num_views)
                .unsqueeze(0)
                .repeat(elevations.shape[0], 1)
            )
            k_nearest_indices[:-add_global_k, -add_global_k:] = global_part[
                :-add_global_k, :add_global_k
            ]

        return k_nearest_indices

    def __len__(self):
        return len(self.obj_paths)

    def __getitem__(self, index):
        num_views = self.num_views
        index = index % len(self.obj_paths)
        obj_path = self.obj_paths[index]
        meta_fp = os.path.join(obj_path, "meta.json")
        with open(meta_fp, "r") as f:
            meta = json.load(f)

        fov = meta["camera_angle_x"]
        image_size = self.img_wh[0]
        focal_length = 0.5 * image_size / np.tan(0.5 * fov)
        intrinsics = np.array(
            [
                [focal_length, 0, image_size/2],
                [0, focal_length, image_size/2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsics = torch.from_numpy(intrinsics)

        # sample or select view ids in a mode
        num_views_all = len(meta["locations"])
        view_ids = self.sample_views(self.sample_views_mode, num_views_all, num_views)

        if self.add_global_k is not None:
            global_view_ids = np.array(self.global_view_ids, dtype=int)
            view_ids = np.concatenate((view_ids, global_view_ids[: self.add_global_k]))

            num_views += self.add_global_k

        locations = [meta["locations"][i] for i in view_ids]

        # load images, elevations, azimuths, c2w_matrixs
        bg_color = self.get_bg_color(self.bg_color)
        img_paths, img_tensors, elevations, azimuths, c2w_matrixs = [], [], [], [], []
        for loc in locations:
            img_path = os.path.join(obj_path, loc["frames"][0]["name"])
            img = self.load_image(img_path, bg_color, return_type="pt").permute(2, 0, 1)
            img_tensors.append(img)
            img_paths.append(img_path)
            elevations.append(loc["elevation"])
            azimuths.append(loc["azimuth"])
            c2w_matrixs.append(np.array(loc["transform_matrix"]))


        # concat and stack
        img_tensors = torch.stack(img_tensors, dim=0).float()  # (Nv, 3, H, W)
        elevations = torch.tensor(elevations).float()  # (Nv,)
        azimuths = torch.tensor(azimuths).float()  # (Nv,)
    
        c2w_matrixs = torch.tensor(c2w_matrixs).float()  # (Nv, 4, 4)
        c2w_matrixs[:, :3, 1:3] *= -1

        w2c_matrixs = torch.inverse(c2w_matrixs)  # (Nv, 4, 4)
        d_elevations = (elevations - elevations[0:1]).reshape(-1, 1)
        d_azimuths = (azimuths - azimuths[0:1]).reshape(-1, 1) % (2 * math.pi)
        distances = torch.zeros_like(d_elevations)
        img_tensors = img_tensors.permute(0, 2, 3, 1)
        return {
            "target_image": img_tensors[1:],
            "input_image": img_tensors[0],
            "elevations": d_elevations[1:],
            "azimuths": d_azimuths[1:],
            "distances": distances[1:],
            "intrinsics": repeat(intrinsics, "i j -> n i j", n=num_views-1),
            "w2c_matrixs": w2c_matrixs[1:],
        }


class MultiViewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

        self.num_workers = num_workers if num_workers else train_batch_size * 2

    def prepare_data(self) -> None:
        # TODO: check if data is available
        pass
    
    def _dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if isinstance(self.data_val, ListConfig):
            return [self._dataloader(dataset, self.hparams.val_batch_size, False) for dataset in self.data_val]
        elif isinstance(self.data_val, DictConfig):
            return [self._dataloader(dataset, self.hparams.val_batch_size, False) for _, dataset in self.data_val.items()]
        else:
            return self._dataloader(self.data_val, self.hparams.val_batch_size, False)

    def test_dataloader(self) -> DataLoader[Any]:
        if isinstance(self.data_test, ListConfig):
            return [self._dataloader(dataset, self.hparams.test_batch_size, False) for dataset in self.data_test]
        elif isinstance(self.data_test, DictConfig):
            return [self._dataloader(dataset, self.hparams.test_batch_size, False) for _, dataset in self.data_test.items()]
        else:
            return self._dataloader(self.data_test, self.hparams.test_batch_size, False)


if __name__ == "__main__":
    from torchvision.utils import save_image

    dataset = MultiViewDataset(
        root_dir="/mnt/pfs/data/render_lvis_hzh",
        num_views=16,
        bg_color="white",
        img_wh=(256, 256),
        sample_views_mode="random",
        caption_path="/mnt/pfs/data/render_lvis_hzh/caption_val.txt",
    )

    sample = dataset[0]
    save_image((sample["images"] + 1.0) / 2, "temp.png", nrow=8, value_range=(0, 1))