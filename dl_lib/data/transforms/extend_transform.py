#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import inspect
from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar

import cv2
import numpy as np
import torch

from .transform_util import to_float_tensor, to_numpy

__all__ = [
    "BlendTransform", "CropTransform", "GridSampleTransform", "HFlipTransform",
    "NoOpTransform", "ScaleTransform", "DistortTransform",
    "BoxJitterTransform", "Transform", "TransformList", "PadTransform",
    "CropPadTransform"
]

# NOTE: to document methods in subclasses, it's sufficient to only document those whose
# implemenation needs special attention.


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of __deterministic__ transformations for
    image and other data structures. "Deterministic" requires that the output of
    all methods of this class are deterministic w.r.t their input arguments. In
    training, there should be a higher-level policy that generates (likely with
    random variations) these transform ops. Each transform op may handle several
    data types, e.g.: image, coordinates, segmentation, bounding boxes. Some of
    them have a default implementation, but can be overwritten if the default
    isn't appropriate. The implementation of each method may choose to modify
    its input data in-place for efficient transformation.
    """
    def _set_attributes(self, params: list = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """
        pass

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """

        pass

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array.
        By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Callable):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(
                str(argspec)))
        setattr(cls, "apply_" + data_type, func)


_T = TypeVar("_T")


# pyre-ignore-all-errors
class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """
    def __init__(self, transforms: list):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name: str):
        """
        Args:
            name (str): name of the attribute.
        """
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError(
            "TransformList object has no attribute {}".format(name))

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        return TransformList(others + self.transforms)


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


class DistortTransform(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """
    def __init__(self, hue, saturation, exposure):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = rand_scale(self.saturation)
        dexp = rand_scale(self.exposure)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.asarray(img, dtype=np.float32) / 255.
        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        img[:, :, 0] = H
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.asarray(img, dtype=np.float32)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """
    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-1))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-2))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords


class BoxJitterTransform(Transform):
    """
    A transofrm that perform gt box jittering without changing the image.
    """
    def __init__(self, p: float = 0.0, ratio: int = 0):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Jitter the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the jittered coordinates.
        """
        if np.random.random() > self.p:
            coords = coords.reshape(-1, 4, 2)
            for coord in coords:
                coord[:, 0] += np.random.randint(-self.ratio, high=self.ratio)
                coord[:, 1] += np.random.randint(-self.ratio, high=self.ratio)
            return coords.reshape(-1, 2)
        else:
            return coords


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class ScaleTransform(Transform):
    """
    Resize the image to a target size.
    """
    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp: str):
        """
        Args:
            h, w (int): original image size.
            new_h, new_w (int): new image size.
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html

        Returns:
            ndarray: resized image(s).
        """
        interp_method = interp if interp is not None else self.interp
        # Option of align_corners is only supported for linear, bilinear,
        # and bicubic.
        if interp_method in ["linear", "bilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None

        float_tensor = torch.nn.functional.interpolate(
            to_float_tensor(img),
            size=(self.new_w, self.new_h),
            mode=interp_method,
            align_corners=align_corners,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class GridSampleTransform(Transform):
    def __init__(self, grid: np.ndarray, interp: str):
        """
        Args:
            grid (ndarray): grid has x and y input pixel locations which are
                used to compute output. Grid has values in the range of [-1, 1],
                which is normalized by the input height and width. The dimension
                is `N x H x W x 2`.
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply grid sampling on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        Returns:
            ndarray: grid sampled image(s).
        """
        interp_method = interp if interp is not None else self.interp
        float_tensor = torch.nn.functional.grid_sample(
            to_float_tensor(img),  # NxHxWxC -> NxCxHxW.
            torch.from_numpy(self.grid),
            mode=interp_method,
            padding_mode="border",
            align_corners=False,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray):
        """
        Not supported.
        """
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply grid sampling on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: grid sampled segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class PadTransform(Transform):
    def __init__(self, dw: list, dh: list, img_value=None, seg_value=None):
        super().__init__()
        assert img_value is not None or seg_value is not None
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Pad the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: padded image(s).
        """
        num_axis = len(img.shape)
        npad = [[0, 0] for x in range(num_axis)]
        if num_axis <= 3:
            npad[:2] = [self.dh, self.dw]
        else:
            npad[-3:-2] = [self.dh, self.dw]
        return np.pad(img,
                      npad,
                      mode='constant',
                      constant_values=self.img_value)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: padded coordinates.
        """
        coords[:, 0] += self.dw[0]
        coords[:, 1] += self.dh[0]
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply pad transform on a list of polygons, each represented by a Nx2 array.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: padded polygons.
        """
        return [self.apply_coords(p) for p in polygons]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: padded segmentation.
        """
        npad = [self.dh, self.dw]
        segmentation = np.pad(segmentation,
                              npad,
                              mode='constant',
                              constant_values=self.seg_value)
        return segmentation


class CropTransform(Transform):
    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            return img[..., self.y0:self.y0 + self.h,
                       self.x0:self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w,
                                self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped,
                              geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class CropPadTransform(Transform):
    def __init__(self,
                 x0: int,
                 y0: int,
                 w: int,
                 h: int,
                 new_w: int,
                 new_h: int,
                 img_value=None,
                 seg_value=None):
        super().__init__()
        self._set_attributes(locals())
        self.crop_trans = CropTransform(x0, y0, w, h)
        pad_width = (self.get_pad_width(w,
                                        new_w), self.get_pad_width(h, new_h))
        self.pad_trans = PadTransform(*pad_width, img_value, seg_value)

    def get_pad_width(self, ori: int, tar: int):
        pad_length = max(tar - ori, 0)
        pad_width = [pad_length // 2, pad_length // 2 + pad_length % 2]
        return pad_width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop and Pad the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped and padded image(s).
        """
        img = self.crop_trans.apply_image(img)
        img = self.pad_trans.apply_image(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped and padded coordinates.
        """
        coords = self.crop_trans.apply_coords(coords)
        coords = self.pad_trans.apply_coords(coords)
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop and pad transform on a list of polygons, each represented by a Nx2 array.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped and padded polygons.
        """
        polygons = self.crop_trans.apply_polygons(polygons)
        polygons = self.pad_trans.apply_polygons(polygons)
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: cropped and padded segmentation.
        """
        segmentation = self.crop_trans.apply_segmentation(segmentation)
        segmentation = self.pad_trans.apply_segmentation(segmentation)
        return segmentation


class BlendTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """
    def __init__(self, src_image: np.ndarray, src_weight: float,
                 dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation
