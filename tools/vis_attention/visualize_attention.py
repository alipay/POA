# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import cv2
import random
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import numpy as np
import models

from PIL import Image
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
from loader import ImageFolderInstance
from tqdm import tqdm

matplotlib.use('Agg')

company_colors = [
    (0,160,215), # blue
    (220,55,60), # red
    (245,180,0), # yellow
    (10,120,190), # navy
    (40,150,100), # green
    (135,75,145), # purple
]
company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]

mapping = {
    2642: (0, 1, 2),
    700: (4, 5),
    1837: (2, 4),
    1935: (0, 1, 4),
    2177: (0,1,2),
    3017: (0,1,2),
    3224: (1, 2, 5),
    3340: (2, 1, 3),
    3425: (0, 1, 2),
    1954: (0,1,2),
    2032: (4,0,5),
    3272: (0,1,2),
    3425: (0,1,2),
    3695: (1,5),
    1791:(1,2,5),
    385 : (1, 5),
    4002: (0,1,2,3)
}

def fix_random_seeds(seed=1024):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def apply_mask2(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    t= 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    mask = (mask - mi) / (ma - mi)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t))+ alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    try:
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    except:
        pass
    fig.savefig(fname)
    # print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large'], help='Architecture.')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the single image.")
    parser.add_argument('--data_path', default='/path/to/imagenet/val/', type=str, help='Path of the images\' folder.')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--show_pics", type=int, default=120)
    parser.add_argument("--remove_chunk_id", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    fix_random_seeds(1024)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = models.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        if args.remove_chunk_id:
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'blocks' in k:
                    names  = k.split('.')
                    names.pop(1)
                    new_state_dict['.'.join(names)] = v
                else:
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict
        msg = model.load_state_dict(new_state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        raise Exception("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def show_attn(img, index=None):
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.get_last_selfattention(img.to(device))

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        prefix = f'id{index}_' if index is not None else ''
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img" + ".png"))
        img = Image.open(os.path.join(args.output_dir, "img" + ".png"))

        attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
        for j in range(nh):
            #fname = os.path.join(args.output_dir, prefix + "attn-head" + str(j) + ".png")
            fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            attns.paste(Image.open(fname), (j * attentions.shape[2], 0))

        #fname = os.path.join(args.output_dir, prefix + "attn.png")
        #attns.save(fname)

        # if args.threshold is not None:
        #     #th_attns = Image.new('RGB', (args.image_size[1] * nh, args.image_size[0]))
        #     image = skimage.io.imread(os.path.join(args.output_dir, "img" + ".png"))
        #     for j in range(nh):
        #         fname = os.path.join(args.output_dir, prefix + "mask_th" + str(args.threshold) + "_head" + str(j) + ".png")
        #         #fname = os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) + ".png")
        #         display_instances(image, th_attn[j], fname=fname, blur=False)
        #         #th_attns.paste(Image.open(fname), (j * args.image_size[1], 0))

        #     #fname = os.path.join(args.output_dir, prefix + "mask_th" + str(args.threshold) + "_head" + str(j) + ".png")
        #     #th_attns.save(fname)

        return attentions, th_attn, img, attns

    def show_attn_color(image, attentions, th_attn, index=None, head=[0,1,2,3,4,5]):
        M = image.max()
        m = image.min()
        span = 64
        image = ((image - m) / (M-m)) * span + (256 - span)
        image = image.mean(axis=2)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        for j in head:
            m = attentions[j]
            m *= th_attn[j]
            attentions[j] = m
        mask = np.stack([attentions[j] for j in head])
        
        blur = False
        contour = False
        alpha = 1
        figsize = tuple([i / 100 for i in args.image_size])
        fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.gca()

        if len(mask.shape) == 3:
            N = mask.shape[0]
        else:
            N = 1
            mask = mask[None, :, :]

        # AJ
        for i in range(N):
            mask[i] = mask[i] * ( mask[i] == np.amax(mask, axis=0))
        a = np.cumsum(mask, axis=0)
        for i in range(N):
            mask[i] = mask[i] * (mask[i] == a[i])
        
        # if imid == 3340:
        #     tmp = company_colors[2]
        #     company_colors[2] = company_colors[1]
        #     company_colors[1] = tmp
        colors = company_colors[:N]

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        margin = 0
        ax.set_ylim(height + margin, -margin)
        ax.set_xlim(-margin, width + margin)
        ax.axis('off')
        masked_image = 0.1*image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            _mask = mask[i]
            if blur:
                _mask = cv2.blur(_mask,(10,10))
            # Mask
            masked_image = apply_mask2(masked_image, _mask, color, alpha)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            if contour:
                padded_mask = np.zeros(
                    (_mask.shape[0] + 2, _mask.shape[1] + 2))#, dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        ax.axis('image')
        fname = os.path.join(args.output_dir, "attn_color.png")
        fig.savefig(fname)
        attn_color = Image.open(fname)

        return attn_color

    if args.image_path is not None and os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)
        attentions, th_attn, pic_i, pic_attn = show_attn(img)
        pic_attn_color = show_attn_color(img.permute(1, 2, 0).cpu().numpy(), attentions, th_attn)
        final_pic = Image.new('RGB', (pic_i.size[1] * 2 + pic_attn.size[0], pic_i.size[1]))
        final_pic.paste(pic_i, (0, 0))
        final_pic.paste(pic_attn_color, (pic_i.size[1], 0))
        final_pic.paste(pic_attn, (pic_i.size[1] * 2, 0))
        final_pic.save(os.path.join(args.output_dir, f"attn.png"))
    
    else:
        train_dataloader = DataLoader(
            ImageFolderInstance(args.data_path, transform=transform), 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=16, 
            pin_memory=True)

        cnt = 0
        for data in tqdm(train_dataloader):
            img, _, idx = data
            for i in range(img.size(0)):
                attentions, th_attn, pic_i, pic_attn = show_attn(img[i:i+1], index=idx[i].item())
                pic_attn_color = show_attn_color(img[i].permute(1, 2, 0).cpu().numpy(), attentions, th_attn, index=idx[i].item())
                final_pic = Image.new('RGB', (pic_i.size[1] * 2 + pic_attn.size[0], pic_i.size[1]))
                final_pic.paste(pic_i, (0, 0))
                final_pic.paste(pic_attn_color, (pic_i.size[1], 0))
                final_pic.paste(pic_attn, (pic_i.size[1] * 2, 0))
                final_pic.save(os.path.join(args.output_dir, f"idx{idx[i].item()}_attn.png"))
                cnt += 1
                if cnt == args.show_pics:
                    exit()
