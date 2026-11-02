# DTrC-Net with the Transformer branch at the official DeiT-tiny width (192-d,
# 3 heads), initialised from the public ImageNet-1k distilled checkpoint.
#
# Why this variant exists: the published DTrC-Net uses a 256-d/8-head DeiT that
# the authors trained themselves on ImageNet-2012 at 256x256 and never released
# (their repo ships only the dataset link). `dtrcnet.py` therefore trained that
# branch from random weights on every dataset -- the vendored loader fell back
# silently -- while the ResNet-34 branch did load ImageNet weights. Keeping the
# comparison fair to the published model requires a pretrained Transformer
# branch, and the only public weights with a compatible architecture are the
# official DeiT-tiny-distilled ones (Touvron et al.), so the token width is
# narrowed 256 -> 192 and the four CTCNet fusion inputs shrink accordingly.
# Everything else (ResNet-34, PPM, SE, RPM/FAM, decoder, losses) is unchanged.
model = dict(
    type='DTrCNetSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    resnet_pretrained=True,
    deit_pretrained=True,
    transformer='public192',
    # Relative to the training cwd (repository root). The loader raises if the
    # file is missing; do not point this at a path that may not exist.
    deit_weights='pretrained/deit_tiny_distilled_patch16_224.fb_in1k.safetensors',
    bce_loss_weight=1.0,
    dice_loss_weight=5.0,
    threshold=0.5,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'dtrcnet_pubdeit'
