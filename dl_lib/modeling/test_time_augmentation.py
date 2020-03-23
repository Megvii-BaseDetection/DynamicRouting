# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from itertools import count

import numpy as np
import torch
from torch import nn

from dl_lib.data.detection_utils import read_image
from dl_lib.data.transforms import ResizeShortestEdge

__all__ = ["DatasetMapperTTA", "SemanticSegmentorWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """
    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a detection dataset dict

        Returns:
            list[dict]:
                a list of dataset dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
        """
        ret = []
        if "image" not in dataset_dict:
            numpy_image = read_image(dataset_dict["file_name"],
                                     self.image_format)
        else:
            numpy_image = dataset_dict["image"].permute(
                1, 2, 0).numpy().astype("uint8")
        for min_size in self.min_sizes:
            image = np.copy(numpy_image)
            tfm = ResizeShortestEdge(min_size,
                                     self.max_size).get_transform(image)
            resized = tfm.apply_image(image)
            resized = torch.as_tensor(
                resized.transpose(2, 0, 1).astype("float32"))

            dic = copy.deepcopy(dataset_dict)
            dic["horiz_flip"] = False
            dic["image"] = resized
            ret.append(dic)

            if self.flip:
                dic = copy.deepcopy(dataset_dict)
                dic["horiz_flip"] = True
                dic["image"] = torch.flip(resized, dims=[2])
                ret.append(dic)
        return ret


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SementicSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """
    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SementicSegmentor): a SementicSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _batch_inference(self, batched_inputs):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`SemanticSegmentor.inference`
        """

        outputs = []
        inputs = []
        for idx, input in zip(count(), batched_inputs):
            inputs.append(input)
            if len(inputs
                   ) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(self.model.forward(inputs, ))
                inputs = []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _hflip_sem_seg(self, x):
        y = x.flip(dims=[2])
        return y

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict

        Returns:
            dict: one output dict
        """
        augmented_inputs = self.tta_mapper(input)

        do_hflip = [k.pop("horiz_flip", False) for k in augmented_inputs]
        heights = [k["height"] for k in augmented_inputs]
        widths = [k["width"] for k in augmented_inputs]
        assert (
            len(set(heights)) == 1 and len(set(widths)) == 1
        ), "Augmented version of the inputs should have the same original resolution!"

        # 1. Segment from all augmented versions
        # 1.1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 1.2: union the results
        for idx, output in enumerate(outputs):
            if do_hflip[idx]:
                output["sem_seg"] = self._hflip_sem_seg(output["sem_seg"])
        all_pred_masks = torch.stack([o["sem_seg"] for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        output = outputs[0]
        output["sem_seg"] = avg_pred_masks
        return output
