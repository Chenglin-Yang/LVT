_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='PanopticFPN',
    backbone = dict(
        _delete_=True,
        type='lvt',
        with_cls_head = False, # classification/downstream tasks
        rasa_cfg = dict(
            atrous_rates= [1,3,5], # None, [1,3,5]
            act_layer= 'nn.SiLU(True)',
            init= 'kaiming',
            r_num = 2,
        ), # rasa setting
        init_cfg=dict(
            checkpoint='pretrained/lvt_imagenet_pretrained.pth.tar',
        ),
    ),
    
    neck=dict(
        type='FPN',
        in_channels=[64, 64, 160, 256],
        out_channels=96,
        num_outs=5),
    
    rpn_head=dict(
        type='RPNHead',
        in_channels=96,
        feat_channels=96,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=96,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=96,
            fc_out_channels=160,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=96,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=96,
            conv_out_channels=96,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    
    semantic_head=dict(
        type='PanopticFPNHead',
        num_classes=54,
        in_channels=96,
        inner_channels=48,
        start_level=0,
        num_upsample = [0,1,2,3],
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=12, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)),
    
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=80,
        num_stuff_classes=53),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096)),
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, weight_decay=0.0001)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

custom_hooks = []
