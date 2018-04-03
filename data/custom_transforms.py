__author__ = 'lamarrem'
import os
import numpy as np
import data.helpers as helpers

# Transformation class
class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self,
                 target,
                 relax=0,
                 zero_pad=False):

        self.target = target
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = self.target.numpy()
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        _img = sample.numpy()
        _crop_img = []
        _crop_gt = []
        # Crop the target
        if _img.ndim == 2:
            _img = np.expand_dims(_img, axis=-1)
        for k in range(0, _target.shape[-1]):
            _tmp_img = _img[..., k]
            _tmp_target = _target[..., k]
            if np.max(_target[..., k]) == 0:
                _crop_gt.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
            else:
                _crop_gt.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
        # Crop the img
        for k in range(0, _target.shape[-1]):
            print(k)
            if np.max(_target[..., k]) == 0:
                print('B')
                _crop_img.append(np.zeros(_img.shape, dtype=_img.dtype))
                print('C')
            else:
                print('D')
                _tmp_target = _target[..., k]
                _crop_img.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
                print('E')
        return _crop_img.from_numpy(), _crop_gt.from_numpy()

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'