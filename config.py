import argparse
import json
import os

import ml_collections



parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--num_works", type=int, default=0)
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--runs_dir', default='./runs', help='runs dir')

# Data
parser.add_argument("--data", type=str, default="MoNuSeg", choices=('MoNuSeg', 'GLySAC','MoNuSAC'))
parser.add_argument("--train_data_dir", type=str, default=r"D:\fdw\data\Gdata\GLySAC\Train\patch\image_patches")
parser.add_argument("--val_data_dir", type=str, default=r"D:\fdw\data\Gdata\GLySAC\Val\patch\image_patches")
parser.add_argument("--test_data_dir", type=str, default=r"D:\fdw\data\Gdata\GLySAC\Test\patch\image_patches")
parser.add_argument("--val_img", type=str, default='')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seed', type=int, default=1234)

# Model
parser.add_argument("--initial_filter_size", type=int, default=32)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--n_labels", type=int, default=4)
parser.add_argument("--n_channels", type=int, default=3)
parser.add_argument("--in_channels", type=int, default=64)
parser.add_argument("--vis", type=bool, default=False)
parser.add_argument("--test_model", type=str,
                    default=r"D:\fdw\code\CausalCellSegmenter\results\train\supervised_54_6_2025-03-24_11-53-18\model\supervised_54_6_2025-03-24_11-53-18.pth")
# Train
parser.add_argument("--restart", default=False, action='store_true')
parser.add_argument("--pretrained_model_path", type=str,
                    default='results/train/MoNuSAC/supervised_53_6_2024-05-20_00-15-48/model/supervised_53_6_2024-05-20_00-15-48.pth')
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--min_lr", type=float, default=1e-7)
parser.add_argument("--T_max", type=int, default=100)
parser.add_argument("--eta_min", type=float, default=1e-6)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='rmsprop', choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--do_contrast", default=True, action='store_true')
parser.add_argument("--lr_scheduler", type=str, default='cos')
parser.add_argument("--vis_frequency", type=int, default=10)
parser.add_argument("--print_frequency", type=int, default=1)
parser.add_argument("--visualize_path", type=str, default='visualize_val')
parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--early_stopping_patience", type=int, default=150)
parser.add_argument("--logger_path", type=str, default='')
parser.add_argument("--tensorboard", type=bool, default=True)
parser.add_argument("--out_chs", nargs='+', type=int, default=[64, 128, 256, 512, 512])
parser.add_argument("--comdice", type=float, default=0.95)
parser.add_argument("--comce", type=float, default=1)
parser.add_argument("--comfl", type=float, default=0.05)

# Contrast Loss
parser.add_argument("--temp", type=float, default=0.05)
parser.add_argument("--slice_threshold", type=float, default=0.0001)
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')

# stable net
parser.add_argument('--epochb', type=int, default=54, help='number of epochs to balance')
parser.add_argument('--num_f', type=int, default=6, help='number of fourier spaces')
parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')
parser.add_argument('--lambdap', type=float, default=70, help='weight decay for weight1 ')
parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--lambda_decay_rate', type=float, default=1, help='ratio of epoch for lambda to decay')
parser.add_argument('--lambda_decay_epoch', type=int, default=20, help='number of epoch for lambda to decay')
parser.add_argument('--min_lambda_times', type=float, default=0.01, help='number of global table levels')
parser.add_argument('--first_step_cons', type=float, default=1, help='constrain the weight at the first step')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
parser.add_argument('--feature_dim', type=int, default=3136, help='the dim of each feature')






def get_config():
    config = parser.parse_args()

    config.transformer = ml_collections.ConfigDict()
    config.data_dir = os.path.expanduser(config.data_dir)
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.KV_size = 960

    config.expand_ratio = 4
    config.base_channel = 64  # base channel of U-Net

    return config
