# HRDA UDA baseline for Crack500 -> UAV crack segmentation.
#
# Third-party HRDA code is kept under external/HRDA. This file only defines the
# project-specific binary crack dataset and the unified 360x360 comparison
# protocol.

_base_ = [
    "../../external/HRDA/configs/_base_/default_runtime.py",
    "../../external/HRDA/configs/_base_/models/daformer_sepaspp_mitb5.py",
    "../../external/HRDA/configs/_base_/uda/dacs.py",
    "../../external/HRDA/configs/_base_/schedules/adamw.py",
    "../../external/HRDA/configs/_base_/schedules/poly10warm.py",
]

seed = 42
n_gpus = 1
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

dataset_type = "CustomDataset"
data_root = "generated/hrda/crack500_to_uav"
classes = ("background", "crack")
palette = [[0, 0, 0], [255, 255, 255]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
crop_size = (360, 360)
train_img_scale = (360, 360)
test_img_scale = (360, 360)

# HRDA needs crop coordinates divisible by the output-stride alignment factor.
hr_crop_size = (176, 176)

source_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=train_img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

target_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", img_scale=train_img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=test_img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

common_dataset_kwargs = dict(
    type=dataset_type,
    img_suffix=".png",
    seg_map_suffix=".png",
    classes=classes,
    palette=palette,
    ignore_index=255,
    reduce_zero_label=False,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type="UDADataset",
        source=dict(
            **common_dataset_kwargs,
            data_root=f"{data_root}/source",
            img_dir="img_dir/train",
            ann_dir="ann_dir/train",
            split="splits/train.txt",
            pipeline=source_train_pipeline,
        ),
        target=dict(
            **common_dataset_kwargs,
            data_root=f"{data_root}/target",
            img_dir="img_dir/train",
            split="splits/train.txt",
            pipeline=target_train_pipeline,
            test_mode=True,
        ),
        rare_class_sampling=dict(min_pixels=100, class_temp=0.01, min_crop_ratio=0.5),
        sync_crop_size=crop_size,
    ),
    val=dict(
        **common_dataset_kwargs,
        data_root=f"{data_root}/target",
        img_dir="img_dir/val",
        ann_dir="ann_dir/val",
        split="splits/val.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        **common_dataset_kwargs,
        data_root=f"{data_root}/target",
        img_dir="img_dir/test",
        ann_dir="ann_dir/test",
        split="splits/test.txt",
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="HRDAEncoderDecoder",
    pretrained="external/HRDA/pretrained/mit_b5.pth",
    decode_head=dict(
        type="HRDAHead",
        single_scale_head="DAFormerHead",
        attention_classwise=True,
        hr_loss_weight=0.1,
        num_classes=2,
    ),
    scales=[1, 0.5],
    hr_crop_size=hr_crop_size,
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode="slide",
        batched_slide=True,
        stride=hr_crop_size,
        crop_size=crop_size,
    ),
)

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[1],
    imnet_feature_dist_scale_min_ratio=0.75,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
)

optimizer_config = None
optimizer = dict(
    lr=6e-5,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)

runner = dict(type="IterBasedRunner", max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric="mIoU")

name = "crack500_to_uav_hrda_mitb5_s0"
exp = "reviewer_uda_baseline"
name_dataset = "crack500_to_uav_360x360"
name_architecture = "hrda176-0.1_daformer_sepaspp_sl_mitb5"
name_encoder = "mitb5"
name_decoder = "hrda176-0.1_daformer_sepaspp_sl"
name_uda = "dacs_a999_fd_crack_rcs0.01"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x1_40k"
