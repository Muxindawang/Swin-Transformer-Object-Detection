from argparse import ArgumentParser
from functools import partial
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    args.device='cpu'

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    H, W, C = 800, 1216, 3
    dummy_input = torch.rand(1, C, H, W, device=args.device)  # 批次维度为1

    # 构造对应的meta信息
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': 'random_data.png',
        'scale_factor': 1.0,
        'flip': False,
        'show_img': dummy_input,  # 可以显示的可视化图像
    }

    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss=False)
    tensor_data = [dummy_input]
    torch.onnx.export(
        model,
        args=tensor_data,
        f="swintracsformer-od.onnx",
        input_names=['input'],
        output_names=['boxes', 'labels', 'masks'],
        export_params=True,
        do_constant_folding=True,
        verbose=True,
        opset_version=14
    )


if __name__ == '__main__':
    main()
