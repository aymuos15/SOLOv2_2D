from .piplines import LoadImageFromFile, LoadAnnotations, Normalize, DefaultFormatBundle, Collect, TestCollect, Pad, Format, ImageToTensor

process_funcs_dict = {'LoadImageFromFile':  LoadImageFromFile,
                      'LoadAnnotations': LoadAnnotations,
                      'Normalize': Normalize,
                      'DefaultFormatBundle': DefaultFormatBundle,
                      'Collect': Collect,
                      'TestCollect': TestCollect,
                      'Pad': Pad,
                      'Format': Format, # This is only in testing anyways
                      'ImageToTensor': ImageToTensor}

# These are in RGB and are for ImageNet
MEANS = (123.675, 116.28, 123.675)
STD = (58.395, 57.12, 58.395)

# ----------------------- DATASETS ----------------------- #
DUMMY_CLASSES = ('circle', 'square')
DUMMY_LABEL_MAP = {1: 1, 2: 2}
DUMMY_LABEL = [1, 2]
# DUMMY_CLASSES = ('square')
# DUMMY_LABEL_MAP = {1: 1}
# DUMMY_LABEL = [1]


class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        return self.name
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# ----------------------- DATASETS ----------------------- #
dummy = Config({
    'name': 'dummy',
    
    'train_prefix': './datasets/dummy_2d/imgs/',
    'train_info': 'annotations.json',
    'trainimg_prefix': '',
    'train_images': './datasets/dummy_2d/imgs/',

    'valid_prefix': './datasets/dummy_2d/',
    'valid_info': 'annotations.json',
    'validimg_prefix': './datasets/dummy_2d/imgs/',
    'valid_images': './datasets/dummy_2d/imgs/',

    'class_names': DUMMY_CLASSES,
    'label_map': DUMMY_LABEL_MAP,
    'label': DUMMY_LABEL,

    'num_classes': len(DUMMY_CLASSES),
})

# ----------------------- BACKBONES ----------------------- #
resnet18_backbone = Config({
    'name': 'resnet18',
    # 'path': './pretrained/resnet18_nofc.pth',
    'path': None,
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

#fpn config
fpn_base = Config({
    'in_channels': [64, 128, 256, 512],
    'out_channels': 256,
    'start_level': 0,
    'num_outs': 5,
})

# ----------------------- TRAIN PIPELINE ----------------------- #
train_pipeline = [
    dict(type='LoadImageFromFile'),                                 
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),         
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),                  
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], 
                         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg')),   
]

# ----------------------- TRAIN CONFIG ----------------------- #
train_config = Config({
    'imgs_per_gpu': 48,
    'workers_per_gpu': 2,
    'num_gpus': 1,
    'train_pipeline': train_pipeline,
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.01, step=[27, 33]),
    'optimizer': dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),  
    'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),
    'total_epoch': 10,
    'epoch_iters_start': 1,
})

# ----------------------- TEST PIPELINE ----------------------- #
test_pipeline = [
    dict(type='LoadImageFromFile'),
]

# ----------------------- TEST CONFIG ----------------------- #
test_config = Config({
    'test_pipeline': test_pipeline,
    'test_cfg': dict(
                nms_pre=500,
                score_thr=0.1,
                mask_thr=0.5,
                update_thr=0.05,
                kernel='gaussian',  # gaussian/linear
                sigma=2.0,
                max_per_img=30)
})

# ----------------------- SOLO v2.0 CONFIGS ----------------------- #
solov2_base_config = dummy.copy({
    'name': 'solov2_base',
    'backbone': resnet18_backbone,
    'dataset': dummy,
    'train_config': train_config,
    'test_config': test_config,
})

cfg = solov2_base_config.copy()

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)