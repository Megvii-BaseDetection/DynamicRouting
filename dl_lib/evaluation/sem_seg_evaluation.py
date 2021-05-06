# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from dl_lib.utils.file_io import PathManager
import cv2
from dl_lib.data import DatasetCatalog, MetadataCatalog
from dl_lib.utils.comm import all_gather, is_main_process, synchronize
from cityscapesscripts.helpers.labels import trainId2label
from .evaluator import DatasetEvaluator


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset_name, distributed, num_classes, ignore_label=255, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

    def reset(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._predictions = []
        self._real_flops = []
        self._expt_flops = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            if "flops" in output:
                flops = output["flops"]
                self._real_flops.append(flops["real_flops"])
                self._expt_flops.append(flops["expt_flops"])
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            # Cityscapes test output
            if 'cityscapes' in self._dataset_name and 'test' in self._dataset_name:
                pred_converg = pred.copy()
                f_name = input["file_name"].split('/')[-1]
                pred_unique = list(np.unique(pred_converg))
                pred_unique.sort(reverse=True)
                for i in range(len(pred_unique)):
                    pred_converg[pred_converg == int(pred_unique[i])] \
                        = trainId2label[int(pred_unique[i])].id
                save_dir = os.path.join(self._output_dir, 'test_dir')
                if not os.path.exists(save_dir):
                    PathManager.mkdirs(save_dir)

                cv2.imwrite(os.path.join(save_dir, f_name), pred_converg)

            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
            ).reshape(self._N, self._N)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            self._real_flops = all_gather(self._real_flops)
            self._real_flops = list(itertools.chain(*self._real_flops))
            self._expt_flops = all_gather(self._expt_flops)
            self._expt_flops = list(itertools.chain(*self._expt_flops))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        # add flops calculation
        if len(self._real_flops) > 0 and len(self._expt_flops) > 0:
            self._real_flops = [x.item() for x in self._real_flops]
            self._expt_flops = [x.item() for x in self._expt_flops]

            res["mean_real_flops"] = (sum(self._real_flops) / len(self._real_flops)) / 1e3
            res["max_real_flops"] = max(self._real_flops) / 1e3
            res["min_real_flops"] = min(self._real_flops) / 1e3
            res["mean_expt_flops"] = (sum(self._expt_flops) / len(self._expt_flops)) / 1e3
            res["max_expt_flops"] = max(self._expt_flops) / 1e3
            res["min_expt_flops"] = min(self._expt_flops) / 1e3

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list
