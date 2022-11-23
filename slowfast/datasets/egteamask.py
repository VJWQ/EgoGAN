#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import torch
import torch.utils.data
from torchvision.utils import save_image
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging
# from .datasets import decoder as decoder
from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Egteamask(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.

        if self.mode in ["train", "val"]:
        # if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EGTEAMASK {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = self.cfg.DATA.PATH_TO_SPLIT_DIR
        self._label_path = self.cfg.DATA.PATH_TO_LABEL_DIR
        self._maskpath_f = self.cfg.DATA.PATH_TO_LABEL_F_DIR
        self._maskpath_m = self.cfg.DATA.PATH_TO_LABEL_M_DIR
        self._maskpath_l = self.cfg.DATA.PATH_TO_LABEL_L_DIR

        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._label_mask = []

        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)[0]

                # fi-m-la
                mask_folder_path_f = os.path.join(self._maskpath_f, path)
                mask_folder_path_m = os.path.join(self._maskpath_m, path)
                mask_folder_path_l = os.path.join(self._maskpath_l, path)

                label_path_f = os.listdir(mask_folder_path_f)
                label_path_m = os.listdir(mask_folder_path_m)
                label_path_l = os.listdir(mask_folder_path_l)

                img_f = cv2.imread(os.path.join(mask_folder_path_f, label_path_f[-1]), cv2.IMREAD_GRAYSCALE)
                img_m = cv2.imread(os.path.join(mask_folder_path_m, label_path_m[-1]), cv2.IMREAD_GRAYSCALE)
                img_l = cv2.imread(os.path.join(mask_folder_path_l, label_path_l[-1]), cv2.IMREAD_GRAYSCALE)

                img_f = cv2.resize(img_f, (340,256))
                img_m = cv2.resize(img_m, (340,256))
                img_l = cv2.resize(img_l, (340,256))

                label = np.concatenate((img_f, img_m, img_l), axis=0)
                
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, path+'.mp4')
                    )

                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = path

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load EGTEAMASK split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing EGTEAMASK dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]    # 256
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]    # 320
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE           # 224
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            spatial_sample_index = 1
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]    # 256
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]    # 320
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE           # 224
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )


        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)

            # fi-m-la
            label = torch.FloatTensor(self._labels[index])
            label = torch.reshape(label, (3, 256,340))
            label = torch.unsqueeze(label, 0)   # torch.Size([1, 2, 256, 340])
            tmp = torch.ones([1, 1, 256, 340])
            label = torch.cat((label, tmp), dim=1)
            label = label.repeat(1,2,1,1)
            comb = torch.cat((frames, label), dim=0)   # torch.Size([1, 8, 256, 340])
            comb = utils.spatial_sampling(
                comb,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            # fi-m-la
            frames, label = torch.split(comb, [3, 1]) # torch.Size([3, 8, 224, 224]), torch.Size([1, 8, 224, 224])
            label, _ = torch.split(label, [3, 5], dim=1)    # torch.Size([1, 3, 224, 224])
            label = label/255
            mask = torch.ones([1, 3, 224, 224])
            bg_mask = mask.sub(label)   # torch.Size([1, 3, 224, 224])
            bg_label = torch.cat((label, bg_mask), dim=0)   # torch.Size([2, 3, 224, 224])
            meta = self. _video_meta[index]
            frames = utils.pack_pathway_output(self.cfg, frames)

            return frames, bg_label, index, meta
            
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
