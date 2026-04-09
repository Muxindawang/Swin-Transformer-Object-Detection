# tools/prune_resnet50_simple.py
import argparse
import torch
from mmcv import Config
from mmdet.models.necks import FPN
from mmdet.models import build_detector
import torch_pruning as tp


def main():
    parser = argparse.ArgumentParser(description='Prune ResNet-50 in Mask R-CNN (32x aligned)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file (.pth)')
    parser.add_argument('--pruning-ratio', type=float, default=0.25,  # ← 关键：用 0.25
                        help='Use 0.25 for 32x alignment: 256->192, 512->384, etc.')
    parser.add_argument('--out-dir', type=str, default='pruned_r50_32x.pth')
    args = parser.parse_args()

    # ----------------------------
    # 1. Load model
    # ----------------------------
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval().cpu()

    backbone = model.backbone
    dummy_input = torch.randn(1, 3, 224, 224)

    # ----------------------------
    # 2. Use standard pruner (auto handles residual connections!)
    # ----------------------------
    pruner = tp.pruner.MagnitudePruner(
        model=backbone,
        example_inputs=dummy_input,
        importance=tp.importance.MagnitudeImportance(),
        pruning_ratio=args.pruning_ratio,
        global_pruning=False,
        iterative_steps=1,
        root_module_types=(torch.nn.Conv2d,),  # 只以 Conv 为剪枝单元
        # 关键：让 pruner 自动找到并剪枝后续的 BN
        forward_fn=lambda m, x: m(x),
        output_transform=None,
    )

    print("[INFO] Before pruning:")
    base_macs, base_params = tp.utils.count_ops_and_params(backbone, dummy_input)
    print(f"  MACs: {base_macs / 1e9:.2f} G, Params: {base_params / 1e6:.2f} M")

    pruner.step()  # ← torch-pruning automatically handles downsample and BN!

    print("[INFO] After pruning:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(backbone, dummy_input)
    print(f"  MACs: {pruned_macs / 1e9:.2f} G, Params: {pruned_params / 1e6:.2f} M")

    # ----------------------------
    # 3. Verify output channels are 32x
    # ----------------------------
    with torch.no_grad():
        outputs = backbone(dummy_input)
    new_channels = [out.shape[1] for out in outputs]
    print(f"[INFO] New output channels: {new_channels}")
    for i, ch in enumerate(new_channels):
        assert ch % 32 == 0, f"Stage {i} channel {ch} is not multiple of 32!"
    print("✅ All channels are multiples of 32 — DLA compatible!")

    # ----------------------------
    # 4. Rebuild neck (FPN) to match new channels
    # ----------------------------
    old_neck_cfg = cfg.model.neck
    new_neck_cfg = dict(
        # type='FPN',
        in_channels=new_channels,          # ← 关键：使用新通道数
        out_channels=old_neck_cfg.out_channels,
        num_outs=old_neck_cfg.num_outs)


    model.neck = FPN(**new_neck_cfg)
    print(f"[INFO] Rebuilt neck with in_channels={new_channels}")

    # ----------------------------
    # 5. Save FULL MODEL (not just backbone!)
    # ----------------------------
    torch.save({
        'state_dict': model.state_dict(),     # ← 完整模型！包含 backbone + neck + heads
        'meta': {
            'pruning_ratio': args.pruning_ratio,
            'backbone_channels': new_channels,
            'config': args.config,
            'original_checkpoint': args.checkpoint
        }
    }, args.out_dir)
    print(f"[INFO] Saved FULL pruned model to {args.out_dir}")
    print("Backbone layer2.0.conv1 weight shape:", model.backbone.layer2[0].conv1.weight.shape)

    # ----------------------------
    # 6. OPTIONAL: Directly evaluate the pruned model (no config rebuild!)
    # ----------------------------
    eval_after_prune = True
    if eval_after_prune:
        print("\n[INFO] Starting evaluation on pruned model...")
        from mmdet.datasets import build_dataloader, build_dataset
        from mmdet.apis import single_gpu_test
        from mmcv.parallel import MMDataParallel
        import os

        # Load test config and dataset
        eval_cfg = Config.fromfile(args.config)
        eval_cfg.model.pretrained = None
        eval_cfg.data.test.test_mode = True

        # Build dataset
        dataset = build_dataset(cfg.data.test)
        samples_per_gpu = 1
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # Use the already-pruned model (structure is correct!)
        model.cuda()
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

        # Run inference
        outputs = single_gpu_test(model, data_loader)

        # Evaluate
        eval_kwargs = eval_cfg.get('evaluation', {}).copy()
        eval_kwargs.update(dict(metric=args.eval))
        results = dataset.evaluate(outputs, **eval_kwargs)
        print("\n" + "="*50)
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v}")
        print("="*50)


if __name__ == '__main__':
    main()