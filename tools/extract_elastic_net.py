# Copyright 2024 Ant Group.
import torch
import numpy as np
import argparse
from collections import OrderedDict
from pathlib import Path


def get_args_parser(
    description='Tool for extracting elastic network', parents=None, add_help=True,
):
    parser = argparse.ArgumentParser(
        description=description, parents=parents or [], add_help=add_help,
    )
    parser.add_argument(
        "--intact-ckpt",
        type=str,
        help="Pretrained checkpoint file of ofa ssl",
    )
    parser.add_argument(
        "--block-chunks",
        type=int,
        default=4,
        help="Block chunk number of ViT arch",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default='vit',
        choices=['vit', 'swin'],
        help="Network type",
    )
    parser.add_argument(
        "--intact-net",
        type=str,
        default='large',
        help="Arch name of intact network",
    )
    parser.add_argument(
        "--elastic-nets",
        help="Arch names of elastic networks",
        default=['small', 'base', 'large'],
        nargs="+",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to save extracted checkpoint file",
    )
    return parser.parse_args()


def extract_elastic_vit(args):
    output_dir = Path(args.output_dir)
    if not args.output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
    print('Loading checkpoint file: {} ...'.format(args.intact_ckpt))
    intact_model = torch.load(args.intact_ckpt)['teacher']
    depths = {'small': 12, 'base': 12, 'large': 24}
    max_depth = depths[args.intact_net]
    widths = {'small': 384, 'base': 768, 'large': 1024}
    max_width = widths[args.intact_net]
    for net in args.elastic_nets:
        depth = depths[net]
        width = widths[net]
        linear_w_scale = max_width / width
        elastic_ckpt_name = Path(args.intact_ckpt).stem + f'_{net}.pth'
        elastic_ckpt_file = str(output_dir / elastic_ckpt_name)
        elastic_state = OrderedDict()
        acitvated_block_idx = np.linspace(0, max_depth - 1, depth).astype(np.int64)
        block_id_dict = {x: i for i, x in enumerate(acitvated_block_idx)}
        print(block_id_dict)
        if args.block_chunks > 0:
            block_each_chunk = depth // args.block_chunks
        else:
            block_each_chunk = None
        for k, v in intact_model.items():
            k = k.replace('backbone.', '')
            # cls_token / pos_embed
            if 'cls_token' in k or 'pos_embed' in k:
                v_ = v[:, :, :width]
                elastic_state[k] = v_
                print(k, v.shape, '->', v_.shape)
            # mask_token
            if 'mask_token' in k:
                v_ = v[:, :width]
                elastic_state[k] = v_
                print(k, v.shape, '->', v_.shape)
            # patch_embed
            if 'patch_embed.proj' in k:
                v_ = v[:width]
                elastic_state[k] = v_
                print(k, v.shape, '->', v_.shape)
            # blocks
            if 'blocks' in k:
                block_idx = int(k[k.find('blocks'):].split('.')[1])
                if block_idx not in acitvated_block_idx:
                    continue
                # update block id 
                new_block_id = block_id_dict[block_idx]
                k = k.replace(f'blocks.{block_idx}', f'blocks.{new_block_id}')
                # add chunk id (used in poa)
                if block_each_chunk is not None:
                    chunk_id = new_block_id // block_each_chunk
                    k = k.replace('blocks', 'blocks.{}'.format(chunk_id))
                # norm layer
                if 'norm.' in k:
                    k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight')
                    v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # linear in attn for qkv
                if 'attn.qkv.linear' in k:
                    k_ = k.replace('linear.', '')
                    max_out_features = v.shape[0] // 3
                    idx_outs = []
                    for i in range(3):
                        idx_outs.append(
                            torch.arange(
                                i * max_out_features,
                                i * max_out_features + width,
                            )
                        )
                    idx_out = torch.cat(idx_outs)
                    if 'weight' in k:
                        v_ = v[idx_out, :][:, :width] * linear_w_scale
                    else:
                        v_ = v[idx_out]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # linear in attn for projection
                if 'attn.proj.linear' in k:
                    k_ = k.replace('linear.', '')
                    if 'weight' in k:
                        v_ = v[:width, :width] * linear_w_scale
                    else:
                        v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # layer scale
                if 'gamma' in k:
                    v_ = v[:width]
                    elastic_state[k] = v_
                    print(k, v.shape, '->', v_.shape)
                # mlp layer, mlp raito: 4
                if 'mlp.fc1' in k:
                    k_ = k.replace('linear.', '')
                    if 'weight' in k:
                        v_ = v[:width * 4, :width] * linear_w_scale
                    else:
                        v_ = v[:width * 4]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                if 'mlp.fc2' in k:
                    k_ = k.replace('linear.', '')
                    if 'weight' in k:
                        v_ = v[:width, :width * 4] * linear_w_scale
                    else:
                        v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
            # norm 
            if k.startswith('backbone.norm'):
                k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight')
                v_ = v[:width]
                elastic_state[k_] = v_
                print(k, '->', k_, v.shape, '->', v_.shape)
        #torch.save({'teacher': elastic_state}, elastic_ckpt_file)
        torch.save(elastic_state, elastic_ckpt_file)
                

def extract_elastic_swin(args, remove_dynamic=False):
    output_dir = Path(args.output_dir)
    head_dim = 16
    if not args.output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
    print('Loading checkpoint file: {} ...'.format(args.intact_ckpt))
    intact_model = torch.load(args.intact_ckpt)['teacher']
    depths = {'tiny': 6, 'small': 18, 'base': 18}
    max_depth = depths[args.intact_net]
    widths = {'tiny': 96, 'small': 96, 'base': 128}
    max_width = widths[args.intact_net]
    for net in args.elastic_nets:
        depth = depths[net]
        width_init = widths[net]
        linear_w_scale = max_width / width_init
        elastic_ckpt_name = Path(args.intact_ckpt).stem + f'_{net}.pth'
        elastic_ckpt_file = str(output_dir / elastic_ckpt_name)
        elastic_state = OrderedDict()
        acitvated_block_idx = np.linspace(0, max_depth - 1, depth).astype(np.int64)
        block_id_dict = {x: i for i, x in enumerate(acitvated_block_idx)}
        print(block_id_dict)
        for k, v in intact_model.items():
            # mask_token
            if 'mask_token' in k:
                v_ = v[:, :width_init]
                elastic_state[k] = v_
                print(k, v.shape, '->', v_.shape)
            # patch_embed
            if 'patch_embed' in k:
                k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight') if remove_dynamic else k
                v_ = v[:width_init]
                elastic_state[k] = v_
                print(k, '->', k_, v.shape, '->', v_.shape)
            # blocks of stages
            if 'stages' in k and 'downsample' not in k:
                stage_idx = int(k[k.find('stages'):].split('.')[1])
                block_idx = int(k[k.find('blocks'):].split('.')[1])
                width = width_init * 2 ** stage_idx
                if stage_idx == 2 and block_idx not in acitvated_block_idx:
                    continue
                # update block id 
                if stage_idx == 2:
                    new_block_id = block_id_dict[block_idx]
                    k = k.replace(f'blocks.{block_idx}', f'blocks.{new_block_id}')
                # norm layer
                if 'norm.' in k:
                    k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight') if remove_dynamic else k
                    v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # relative position
                if 'relative_position' in k:
                    num_head = width // head_dim
                    if 'bias_table' in k:
                        v_ = v[:, :num_head]
                    else:
                        v_ = v
                    elastic_state[k] = v_
                    print(k, v.shape, '->', v_.shape)
                # linear in attn for qkv
                if 'attn.w_msa.qkv.linear' in k:
                    k_ = k.replace('linear.', '') if remove_dynamic else k
                    max_out_features = v.shape[0] // 3
                    idx_outs = []
                    for i in range(3):
                        idx_outs.append(
                            torch.arange(
                                i * max_out_features,
                                i * max_out_features + width,
                            )
                        )
                    idx_out = torch.cat(idx_outs)
                    if 'weight' in k:
                        v_ = v[idx_out, :][:, :width] * linear_w_scale
                    else:
                        v_ = v[idx_out]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # linear in attn for projection
                if 'attn.w_msa.proj.linear' in k:
                    k_ = k.replace('linear.', '') if remove_dynamic else k
                    if 'weight' in k:
                        v_ = v[:width, :width] * linear_w_scale
                    else:
                        v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # ffn layer, mlp raito: 4
                if 'ffn.fc1' in k:
                    k_ = k.replace('linear.', '') if remove_dynamic else k
                    if 'weight' in k:
                        v_ = v[:width * 4, :width] * linear_w_scale
                    else:
                        v_ = v[:width * 4]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                if 'ffn.fc2' in k:
                    k_ = k.replace('linear.', '') if remove_dynamic else k
                    if 'weight' in k:
                        v_ = v[:width, :width * 4] * linear_w_scale
                    else:
                        v_ = v[:width]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
            # downsample of stages
            if 'stages' in k and 'downsample' in k:
                stage_idx = int(k[k.find('stages'):].split('.')[1])
                width = width_init * 2 ** stage_idx
                max_in_features = v.shape[-1] // 4
                idx_ins = []
                for i in range(4):
                    idx_ins.append(
                        torch.arange(
                            i * max_in_features,
                            i * max_in_features + width,
                        )
                    )
                idx_in = torch.cat(idx_ins)
                # reduction
                if 'reduction' in k:
                    k_ = k.replace('linear.', '') if remove_dynamic else k
                    v_ = v[:width * 2, :][:, idx_in] * linear_w_scale
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
                # norm 
                if 'norm' in k:
                    k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight') if remove_dynamic else k
                    v_ = v[idx_in]
                    elastic_state[k_] = v_
                    print(k, '->', k_, v.shape, '->', v_.shape)
            # norm 
            if k.startswith('backbone.norm'):
                stage_idx = int(k.split('.')[1].replace('norm', ''))
                k_ = k.replace('norm.bias', 'bias').replace('norm.weight', 'weight') if remove_dynamic else k
                width = width_init * 2 ** stage_idx
                v_ = v[:width]
                elastic_state[k_] = v_
                print(k, '->', k_, v.shape, '->', v_.shape)
        #torch.save({'teacher': elastic_state}, elastic_ckpt_file)
        torch.save(elastic_state, elastic_ckpt_file)


def extract_elastic_net(args):
    if args.arch == 'vit':
        extract_elastic_vit(args)
    if args.arch == 'swin':
        extract_elastic_swin(args)


if __name__ == '__main__':
    args = get_args_parser()
    extract_elastic_net(args)



