import os.path as osp
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional

import cv2
import torch
import numpy as np
import pycocotools.mask as maskUtils

from .data_container import DataContainer as DC
from .compose import Compose
from .imgutils import impad, impad_to_multiple

@dataclass
class BaseTransform:
    """Base class for all transforms."""
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to results."""
        return results

def to_tensor(data):
    """Convert objects to torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        # Ensure float arrays use float32 not float64
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, (int, float)):
        tensor_type = torch.LongTensor if isinstance(data, int) else torch.FloatTensor
        return tensor_type([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@dataclass
class LoadImageFromFile(BaseTransform):
    """Load an image from file."""
    to_float32: bool = False
    color_type: str = 'color'
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Handle direct image path (used in inference)
        if 'img' in results and isinstance(results['img'], str):
            filename = results['img']
        # Handle standard format with img_info (used in training)
        elif 'img_info' in results:
            if results['img_prefix'] is not None:
                filename = osp.join(results['img_prefix'], results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
        else:
            raise ValueError("Cannot find image path in results. Need either 'img' as string or 'img_info'.")
            
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.to_float32:
            img = img.astype(np.float32)
            
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

@dataclass
class LoadAnnotations(BaseTransform):
    """Load annotations for object detection."""
    with_bbox: bool = True
    with_label: bool = True
    with_mask: bool = False
    poly2mask: bool = True
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        ann_info = results['ann_info']
        
        if self.with_bbox:
            results['gt_bboxes'] = ann_info['bboxes']
            results['bbox_fields'] = results.get('bbox_fields', []) + ['gt_bboxes']
            
            if 'bboxes_ignore' in ann_info:
                results['gt_bboxes_ignore'] = ann_info['bboxes_ignore']
                results['bbox_fields'].append('gt_bboxes_ignore')
        
        if self.with_label:
            results['gt_labels'] = ann_info['labels']
        
        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = ann_info['masks']
            
            if self.poly2mask:
                gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
                
            results['gt_masks'] = gt_masks
            results['mask_fields'] = results.get('mask_fields', []) + ['gt_masks']
        
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

@dataclass
class Normalize(BaseTransform):
    """Normalize the image."""
    mean: Union[List[float], Tuple[float, ...]]
    std: Union[List[float], Tuple[float, ...]]
    to_rgb: bool = True
    
    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        for key in results.get('img_fields', ['img']):
            img = results[key].copy().astype(np.float32)
            
            if self.to_rgb:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            
            mean = self.mean.reshape(1, 1, -1).astype(np.float32)
            std = self.std.reshape(1, 1, -1).astype(np.float32)
            
            img = ((img - mean) / std).astype(np.float32)
            results[key] = img
            
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

@dataclass
class Pad(BaseTransform):
    """Pad the image & mask."""
    size: Optional[Tuple[int, int]] = None
    size_divisor: Optional[int] = None
    pad_val: int = 0
    
    def __post_init__(self):
        assert self.size is not None or self.size_divisor is not None
        assert self.size is None or self.size_divisor is None
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Pad image
        if self.size is not None:
            padded_img = impad(results['img'], shape=self.size, pad_val=self.pad_val)
        else:
            padded_img = impad_to_multiple(results['img'], self.size_divisor, pad_val=self.pad_val)
            
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        
        # Pad masks
        pad_shape = padded_img.shape[:2]
        for key in results.get('mask_fields', []):
            padded_masks = [impad(mask, shape=pad_shape, pad_val=self.pad_val) for mask in results[key]]
            if padded_masks:
                results[key] = np.stack(padded_masks, axis=0)
            else:
                results[key] = np.empty((0,) + pad_shape, dtype=np.uint8)
                
        # Pad segmentations
        for key in results.get('seg_fields', []):
            results[key] = impad(results[key], pad_shape, pad_val=self.pad_val)
            
        return results

@dataclass
class ImageToTensor(BaseTransform):
    """Convert image to torch.Tensor."""
    keys: List[str]
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

@dataclass
class DefaultFormatBundle(BaseTransform):
    """Default formatting bundle."""
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Format image
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
                
            # Get shape information before converting to tensor
            img_shape = img.shape
            num_channels = 1 if len(img_shape) < 3 else img_shape[2]
            
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
            
            # Add default meta keys using the shape info we saved
            results.setdefault('pad_shape', (img_shape[0], img_shape[1], num_channels))
            results.setdefault('scale_factor', 1.0)
            results.setdefault('img_norm_cfg', dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        
        # Format bounding boxes and labels
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key in results:
                results[key] = DC(to_tensor(results[key]))
        
        # Format masks
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
            
        return results

@dataclass
class Collect(BaseTransform):
    """Collect data from the loader relevant to the specific task."""
    keys: List[str]
    meta_keys: Tuple[str, ...] = ('filename', 'ori_shape', 'img_shape', 'pad_shape',
                                 'scale_factor', 'flip', 'img_norm_cfg')
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        data = {}
        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data['img_metas'] = DC(img_meta, cpu_only=True)
        
        for key in self.keys:
            data[key] = results[key]
        return data

@dataclass
class TestCollect(Collect):
    """Test time collect for inference."""
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        data = {}
        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data['img_metas'] = img_meta
        
        for key in self.keys:
            data[key] = results[key]
        return data

@dataclass
class Format(BaseTransform):
    """Test-time augmentation without scaling."""
    transforms: List[Any]
    
    def __post_init__(self):
        self.transforms = Compose(self.transforms)
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        _results = results.copy()
        output = self.transforms(_results)
        
        # Wrap the single output in a list format expected by the model
        aug_data_dict = {key: [output[key]] for key in output}
        return aug_data_dict