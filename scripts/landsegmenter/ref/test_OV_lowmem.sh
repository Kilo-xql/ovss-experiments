export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=$1

test_weight_path=checkpoints/LandSegmenter-aux-epoch=49.ckpt

python -u ProxyCLIP/test_from_lmdb_all.py \
    --test_dataset_names $2 \
    --test_batch_sizes 1 \
    --num_workers 0 \
    --use-auxiliary-decoder \
    --dofa-type vit_base_patch16 \
    --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
    --accelerator gpu \
    --amp \
    --test_only \
    --test_weight_path $test_weight_path \
    --vfm_model weak \
    --weak_feature_type encoder_embed \
    --clip_slide_stride 256 \
    --clip_slide_crop 256 \
    "${@:3}"
