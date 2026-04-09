# prune_from_trained_checkpoint.py
import torch
import torch.nn as nn
from mmcv import Config
from mmdet.models import build_detector
from mmdet.models.backbones.swin_transformer import PatchMerging, WindowAttention
# from mmcv.cnn.bricks.transformer import WindowMSA
from mmcv.runner import load_checkpoint
import torch_pruning as tp
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Prune Swin from a trained MMDet checkpoint')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='trained model checkpoint (e.g., epoch_12.pth)')
    parser.add_argument('--pruning-ratio', type=float, default=0.5)
    parser.add_argument('--input-size', type=int, nargs=2, default=[480, 800])
    parser.add_argument('--output', type=str, default='pruned_model.pth')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Step 1: Build full model and load your trained weights
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    print(f"[INFO] Loaded trained model from {args.checkpoint}")

    # Step 2: Extract backbone for pruning
    backbone = model.backbone
    H, W = args.input_size
    assert H % 32 == 0 and W % 32 == 0, "Input size must be divisible by 32"
    dummy_input = torch.randn(1, 3, H, W)

    with torch.no_grad():
        orig_feats = backbone(dummy_input)
        orig_channels = [f.shape[1] for f in orig_feats]
    print(f"[INFO] Original FPN channels: {orig_channels}")
    # ----------------------------
    # 2. Collect qkv layers and num_heads
    # ----------------------------
    num_heads = {}
    for m in backbone.modules():
        if isinstance(m, WindowAttention):
            embed_dim = m.qkv.in_features
            # Swin-Tiny: [96,192,384,768] → [3,6,12,24] heads
            n_h = {96: 3, 192: 6, 384: 12, 768: 24}.get(embed_dim)
            if n_h is None:
                raise ValueError(f"Unknown embed_dim: {embed_dim}")
            num_heads[m.qkv] = n_h
    print(f"[INFO] Found {len(num_heads)} WindowAttention layers for head-aware pruning.")
    # 在 main() 函数中，构建 pruner 之前
    unwrapped_parameters = []
    for name, param in backbone.named_parameters():
        if 'relative_position_bias_table' in name:
            # relative_position_bias_table: [num_distances, num_heads]
            # We prune along the head dimension (dim=1)
            unwrapped_parameters.append( (param, 1) )  # ← (parameter, dim)
    print(f"[INFO] Ignoring {len(unwrapped_parameters)} relative_position_bias_table parameters (prune dim=1).")
    # ----------------------------
    # 3. Custom pruner for PatchMerging
    # ----------------------------
    class QKVPruner(tp.BasePruningFunc):
        def prune_out_channels(self, layer, idxs):
            # Do nothing! Prevent output pruning on qkv
            return layer

        def prune_in_channels(self, layer, idxs):
            # Allow input pruning (from previous layer)
            tp.prune_linear_in_channels(layer, idxs)
            return layer

        def get_out_channels(self, layer):
            return layer.out_features

        def get_in_channels(self, layer):
            return layer.in_features
    class PatchMergingPruner(tp.BasePruningFunc):
        def prune_out_channels(self, layer, idxs):
            tp.prune_linear_out_channels(layer.reduction, idxs)
            return layer

        def prune_in_channels(self, layer, idxs):
            # 获取原始 embed_dim（注意：必须从 norm 层推断！）
            # 因为 reduction.in_features 可能已被剪过
            total_in = layer.norm.normalized_shape[0]  # ← 更可靠！LayerNorm 的 normalized_shape = [4*C]
            assert total_in % 4 == 0, f"Expected total_in divisible by 4, got {total_in}"
            embed_dim = total_in // 4

            # 安全检查：idxs 必须 < embed_dim
            if any(i >= embed_dim for i in idxs):
                raise IndexError(f"idxs {idxs} out of bounds for embed_dim={embed_dim}")

            # 构造 4 份偏移索引
            idxs_rep = []
            for offset in range(4):
                idxs_rep.extend([i + offset * embed_dim for i in idxs])

            # 再次检查不越界
            max_idx = max(idxs_rep)
            if max_idx >= total_in:
                raise IndexError(f"Generated index {max_idx} >= total_in {total_in}")

            # 执行剪枝
            tp.prune_linear_in_channels(layer.reduction, idxs_rep)
            tp.prune_layernorm_out_channels(layer.norm, idxs_rep)
            return layer

        def get_out_channels(self, layer):
            return layer.reduction.out_features

        def get_in_channels(self, layer):
            return layer.norm.normalized_shape[0] // 4  # ← 关键：从 LayerNorm 获取

    # ----------------------------
    # 4. Prune backbone
    # ----------------------------
    base_macs, base_params = tp.utils.count_ops_and_params(backbone, dummy_input)
    print("num_heads mapping:")
    for qkv_layer, n_h in num_heads.items():
        print(f"  qkv {qkv_layer} -> {n_h} heads")
    pruner = tp.pruner.MagnitudePruner(
        model=backbone,
        example_inputs=dummy_input,
        importance=tp.importance.MagnitudeImportance(p=2, group_reduction="mean"),
        pruning_ratio=args.pruning_ratio,
        global_pruning=False,
        iterative_steps=1,
        num_heads=num_heads,  # ✅ Now actually used!
        customized_pruners={PatchMerging: PatchMergingPruner(), nn.Linear: QKVPruner(),},
        root_module_types=(nn.Linear, nn.LayerNorm, PatchMerging),
        ignored_layers=[],
        unwrapped_parameters=unwrapped_parameters,
    )
    try:
        pruner.step()
    except Exception as e:
        print("Error during pruning!")
        # Optional: iterate groups manually to find culprit
        for i, group in enumerate(pruner.DG.get_all_groups()):
            print(f"Group {i}:")
            for dep, idxs in group:
                print(f"  {dep}")
                print(f"    idxs: {idxs[:5]}... (len={len(idxs)})")
                if hasattr(dep.target.module, 'weight'):
                    print(f"    weight shape: {dep.target.module.weight.shape}")
            print()
        raise e
    # 执行迭代式剪枝（按目标 ratio 逐步剪）
    # for g in pruner.step(interactive=True):
    #     #print(g)
    #     g.prune()

    # Update embed_dims in WindowAttention (optional but safe)
    for m in backbone.modules():
        if isinstance(m, WindowAttention):
            m.embed_dims = m.qkv.in_features

    # ----------------------------
    # 5. Rebuild full model
    # ----------------------------
    with torch.no_grad():
        pruned_feats = backbone(dummy_input)
        pruned_channels = [f.shape[1] for f in pruned_feats]
    print(f"[INFO] Pruned FPN channels: {pruned_channels}")

    # Update neck in_channels
    cfg.model.neck.in_channels = pruned_channels
    pruned_model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    pruned_model.backbone = backbone
    pruned_model.CLASSES = model.CLASSES

    # ----------------------------
    # 6. Save checkpoint (compatible with test.py)
    # ----------------------------
    checkpoint = {
        'state_dict': pruned_model.state_dict(),
        'meta': {
            'CLASSES': pruned_model.CLASSES,
            'pruning_ratio': args.pruning_ratio,
            'input_size': args.input_size,
        }
    }
    torch.save(checkpoint, args.output)
    print(f"[INFO] Saved pruned checkpoint to {args.output}")

    # ----------------------------
    # 7. Print stats
    # ----------------------------
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(backbone, dummy_input)
    print(f"[INFO] MACs: {base_macs/1e9:.2f}G → {pruned_macs/1e9:.2f}G "
          f"({(1 - pruned_macs / base_macs) * 100:.1f}% reduction)")
    print(f"[INFO] Params: {base_params/1e6:.2f}M → {pruned_params/1e6:.2f}M")

if __name__ == '__main__':
    main()
