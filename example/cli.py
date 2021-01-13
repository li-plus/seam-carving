import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import seam_carving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('-o', dest='dst', type=str, default='a.jpg')
    parser.add_argument('--keep', type=str, default=None)
    parser.add_argument('--drop', type=str, default=None)
    parser.add_argument('--dw', type=int, default=0)
    parser.add_argument('--dh', type=int, default=0)
    parser.add_argument('--energy', type=str, default='backward',
                        choices=['backward', 'forward'])
    parser.add_argument('--order', type=str, default='width-first',
                        choices=['width-first', 'height-first', 'optimal'])
    args = parser.parse_args()

    try:
        print('Loading source image from {}'.format(args.src))
        src = np.array(Image.open(args.src))

        drop_mask = None
        if args.drop is not None:
            print('Loading drop_mask from {}'.format(args.drop))
            drop_mask = np.array(Image.open(args.drop).convert('L'))

        keep_mask = None
        if args.keep is not None:
            print('Loading keep_mask from {}'.format(args.keep))
            keep_mask = np.array(Image.open(args.keep).convert('L'))

        print('Performing seam carving...')
        start = time.time()
        if drop_mask is not None:
            dst = seam_carving.remove_object(src, drop_mask, keep_mask)
        else:
            src_h, src_w, _ = src.shape
            dst = seam_carving.resize(src, (src_w + args.dw, src_h + args.dh),
                                      args.energy, args.order, keep_mask)
        print('Done at {:.4f} second(s)'.format(time.time() - start))

        print('Saving output image to {}'.format(args.dst))
        Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(dst).save(args.dst)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
