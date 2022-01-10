import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import iou_loss, sigmoid_focal_loss_jit
from cvpods.modeling.meta_arch.fcos import (
    permute_all_cls_and_box_to_N_HWA_K_and_concat,
    Scale
)
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import comm, get_event_storage, log_first_n


def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(
        probs, targets, reduction="none"
    )
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


focal_loss_jit = torch.jit.script(focal_loss)  # type: torch.jit.ScriptModule


class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.FCOS.REG_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.poto_alpha = cfg.MODEL.POTO.ALPHA
        self.center_sampling_radius = cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS
        self.poto_aux_topk = cfg.MODEL.POTO.AUX_TOPK

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_filter = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            gt_classes, gt_shifts_reg_deltas = self.get_ground_truth(
                shifts, gt_instances, box_cls, box_delta, box_filter)
            losses = self.losses(gt_classes, gt_shifts_reg_deltas, box_cls,
                                 box_delta, box_filter)
            gt_classes = self.get_aux_ground_truth(
                shifts, gt_instances, box_cls, box_delta)
            losses.update(self.aux_losses(gt_classes, box_cls))
            return losses
        else:
            results = self.inference(box_cls, box_delta, box_filter, shifts,
                                     images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_shifts_deltas, pred_class_logits,
               pred_shift_deltas, pred_filtering):
        """
        Args:
            For `gt_classes` and `gt_shifts_deltas` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_fitering`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas, pred_filtering = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat(
                pred_class_logits, pred_shift_deltas, pred_filtering,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        pred_class_logits = pred_class_logits.sigmoid() * pred_filtering.sigmoid()

        # logits loss
        loss_cls = focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / max(1.0, num_foreground) * self.reg_weight

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
        }

    def aux_losses(self, gt_classes, pred_class_logits):
        pred_class_logits = cat([
            permute_to_N_HWA_K(x, self.num_classes) for x in pred_class_logits
        ], dim=1).view(-1, self.num_classes)

        gt_classes = gt_classes.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        # logits loss
        loss_cls_aux = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        return {"loss_cls_aux": loss_cls_aux}

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, box_cls, box_delta, box_filter):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        gt_shifts_deltas = []

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_filter = torch.cat([permute_to_N_HWA_K(x, 1) for x in box_filter], dim=1)
        box_cls = box_cls.sigmoid_() * box_filter.sigmoid_()

        num_fg = 0
        num_gt = 0

        for shifts_per_image, targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            quality[~is_in_boxes] = -1

            gt_idxs, shift_idxs = linear_sum_assignment(quality.cpu().numpy(), maximize=True)

            num_fg += len(shift_idxs)
            num_gt += len(targets_per_image)

            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), self.num_classes, dtype=torch.long
            )
            gt_shifts_reg_deltas_i = shifts_over_all_feature_maps.new_zeros(
                len(shifts_over_all_feature_maps), 4
            )
            if len(targets_per_image) > 0:
                # ground truth classes
                gt_classes_i[shift_idxs] = targets_per_image.gt_classes[gt_idxs]
                # ground truth box regression
                gt_shifts_reg_deltas_i[shift_idxs] = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps[shift_idxs], gt_boxes[gt_idxs].tensor
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)

        get_event_storage().put_scalar("num_fg_per_gt", num_fg / num_gt)

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas)

    @torch.no_grad()
    def get_aux_ground_truth(self, shifts, targets, box_cls, box_delta):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
        """
        gt_classes = []

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_cls = box_cls.sigmoid_()

        num_fg = 0
        num_gt = 0

        for shifts_per_image, targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            candidate_idxs = []
            st, ed = 0, 0
            for shifts_i in shifts_per_image:
                ed += len(shifts_i)
                _, topk_idxs = quality[:, st:ed].topk(self.poto_aux_topk, dim=1)
                candidate_idxs.append(st + topk_idxs)
                st = ed
            candidate_idxs = torch.cat(candidate_idxs, dim=1)

            is_in_boxes = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1)
            ).min(dim=-1).values > 0

            candidate_qualities = quality.gather(1, candidate_idxs)
            quality_thr = candidate_qualities.mean(dim=1, keepdim=True) + \
                          candidate_qualities.std(dim=1, keepdim=True)
            is_foreground = torch.zeros_like(is_in_boxes).scatter_(1, candidate_idxs, True)
            is_foreground &= quality >= quality_thr

            quality[~is_in_boxes] = -1
            quality[~is_foreground] = -1

            # if there are still more than one objects for a position,
            # we choose the one with maximum quality
            positions_max_quality, gt_matched_idxs = quality.max(dim=0)

            num_fg += (positions_max_quality != -1).sum().item()
            num_gt += len(targets_per_image)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with quality -1 are treated as background.
                gt_classes_i[positions_max_quality == -1] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            gt_classes.append(gt_classes_i)

        get_event_storage().put_scalar("num_fg_per_gt_aux", num_fg / num_gt)

        return torch.stack(gt_classes)

    def inference(self, box_cls, box_delta, box_filter, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_filter: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_filter = [permute_to_N_HWA_K(x, 1) for x in box_filter]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_filter_per_image = [
                box_filter_per_level[img_idx] for box_filter_per_level in box_filter
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_filter_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_filter, shifts, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_filter (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_filter_i, shifts_i in zip(
                box_cls, box_delta, box_filter, shifts):
            # (HxWxK,)
            box_cls_i = (box_cls_i.sigmoid_() * box_filter_i.sigmoid_()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        if self.nms_type is None:
            # strategies above (e.g. topk_candidates and score_threshold) are
            # useless for POTO, just keep them for debug and analysis
            keep = scores_all.argsort(descending=True)
        else:
            keep = generalized_batched_nms(
                boxes_all, scores_all, class_idxs_all,
                self.nms_threshold, nms_type=self.nms_type
            )
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_filter = self.head(features)
        shifts = self.shift_generator(features)

        results = self.inference(box_cls, box_delta, box_filter, shifts, images)
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_shifts = cfg.build_shift_generator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on
        assert len(set(num_shifts)) == 1, "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_shifts * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        self.max3d = MaxFiltering(in_channels,
                                  kernel_size=cfg.MODEL.POTO.FILTER_KERNEL_SIZE,
                                  tau=cfg.MODEL.POTO.FILTER_TAU)
        self.filter = nn.Conv2d(in_channels,
                                num_shifts * 1,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred,
            self.max3d, self.filter
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            filter (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits = []
        bbox_reg = []
        filter_subnet = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

            filter_subnet.append(bbox_subnet)
        filters = [self.filter(x) for x in self.max3d(filter_subnet)]
        return logits, bbox_reg, filters


class MaxFiltering(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(f, size=x.shape[2:], mode="bilinear")
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs
