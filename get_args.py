import argparse
# import torch
import os


def list_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', '-md', type=str, default='freqdino')
    parser.add_argument('--weights', '-w', type=str, default='freqdino pretrained filename')
    parser.add_argument('--resume', '-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100,
                        help="the iteration number to record train state")
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-4)
    parser.add_argument('--test_mode', '-tm', action='store_true', default=False, help='train or test')
    parser.add_argument('--test_start_epoch', type=int, default=0, help='test start epoch')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--writer', action='store_true', default=False, help='whether use Tensorboard')
    parser.add_argument('--image_size', '-is', type=int, default=512)
    parser.add_argument('--backbone_size', '-bbs', type=str, default='large')
    parser.add_argument('--num_samples', '-ns', type=int, default=20000)
    parser.add_argument('--adapter_interval', '-ai', type=int, default=6, help='interval of adapter')
    parser.add_argument('--hidden_dim_ratio', '-hdr', type=float, default=1,
                        help='hidden dim ratio of adaptor, used as "hidden_dim = in_features * hidden_dim_ratio"')
    parser.add_argument('--adapter_dim_ratio', '-adr', type=float, default=8,
                        help='adapter dim ratio of adaptor, used as "hidden_dim = in_features / adapter_dim_ratio"')
    parser.add_argument('--wavelet', '-w', type=str, default='db4',
                        help='wavelet used on dwt')

    args = parser.parse_args()
    return args
