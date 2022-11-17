# Seam Carving

[![PyPI](https://img.shields.io/pypi/v/seam-carving)](https://pypi.org/project/seam-carving/)
[![Unit Test](https://github.com/li-plus/seam-carving/workflows/Unit%20Test/badge.svg?branch=master)](https://github.com/li-plus/seam-carving/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/li-plus/seam-carving/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/li-plus/seam-carving/branch/master/graph/badge.svg)](https://codecov.io/gh/li-plus/seam-carving)

A super-fast Python implementation of [Seam carving for content-aware image resizing](https://dl.acm.org/doi/abs/10.1145/1275808.1276390), and the forward energy function proposed in [Improved seam carving for video retargeting](https://dl.acm.org/doi/abs/10.1145/1360612.1360615).

With seam carving algorithm, the image could be intelligently resized while keeping the important contents undistorted. The carving process could be further guided, so that an object could be removed from the image without apparent artifacts.

## Installation

Install a stable version from PyPI.

```shell script
pip install seam-carving
```

Or install the latest version from GitHub.

```shell script
pip install git+https://github.com/li-plus/seam-carving.git@master
```

## Quick Start

To resize an image using seam carving:

```python
import numpy as np
from PIL import Image
import seam_carving

src = np.array(Image.open("fig/castle.jpg"))
src_h, src_w, _ = src.shape
dst = seam_carving.resize(
    src,  # source image (rgb or gray)
    size=(src_w - 200, src_h),  # target size
    energy_mode="backward",  # choose from {backward, forward}
    order="width-first",  # choose from {width-first, height-first}
    keep_mask=None,  # object mask to protect from removal
)
Image.fromarray(dst).show()
```

To remove an object from the image:

```python
src = np.array(Image.open("fig/beach.jpg"))
mask = np.array(Image.open("fig/beach_girl.png").convert("L"))
dst = seam_carving.resize(src, drop_mask=mask)
Image.fromarray(dst).show()
```

For more examples, please refer to [example/demo.ipynb](example/demo.ipynb).

## Example Results

### Scaling Up & Down

Resizing along the x-axis using the original backward energy function:

![](docs/basic.jpg)

### Backward Energy vs Forward Energy

Reducing the width using backward & forward energy functions:

![](fig/bench.jpg)

![](docs/backward_vs_forward.jpg)

### Aspect Ratio Change

The image width and height could be changed simultaneously in an optimal seam order, but the order has little effect on the final result. Currently we only support two kinds of seam orders: `width-first` and `height-first`. In `width-first` mode, we remove or insert all vertical seams first, and then the horizontal ones, while `height-first` is the opposite.

![](docs/aspect_ratio.jpg)

### Object Protection

The protected mask is not affected by seam removal and insertion.

![](docs/object_protection.jpg)

### Object Removal

Specify an object mask to remove (red) and a mask to protect (green, optional).

![](docs/object_removal.jpg)

## Benchmarks

We compare the performance of our implementation and other popular Python repos on [castle.jpg](fig/castle.jpg) (600x407). The image is narrowed or widened by 200 pixels using backward energy (BE) or forward energy (FE), respectively. Below is the running time (second) evaluated on a MacBook Pro.

| Methods                                                                                 | BE -200px | BE +200px | FE -200px | FE +200px |
| --------------------------------------------------------------------------------------- | --------- | --------- | --------- | --------- |
| [vivianhylee/seam-carving](https://github.com/vivianhylee/seam-carving)                 | 168.47    | 179.52    | 89.24     | 90.27     |
| [andrewdcampbell/seam-carving](https://github.com/andrewdcampbell/seam-carving)         | 119.47    | 126.44    | 133.29    | 133.73    |
| [sameeptandon/python-seam-carving](https://github.com/sameeptandon/python-seam-carving) | 69.18     | 95.21     | N/A       | N/A       |
| [dharness/seam_carving](https://github.com/dharness/seam_carving)                       | 50.25     | 57.86     | N/A       | N/A       |
| [Ours](https://github.com/li-plus/seam-carving)                                         | **1.03**  | **1.08**  | **1.07**  | **1.17**  |

## References

* Avidan, S., & Shamir, A. (2007). Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers (pp. 10-es). [[paper]](https://dl.acm.org/doi/abs/10.1145/1275808.1276390) [[blog]](https://faculty.idc.ac.il/arik/SCWeb/imret/index.html)
* Rubinstein, M., Shamir, A., & Avidan, S. (2008). Improved seam carving for video retargeting. ACM transactions on graphics (TOG), 27(3), 1-9. [[paper]](https://dl.acm.org/doi/abs/10.1145/1360612.1360615)
* Das, A. (2019). Improved seam carving with forward energy. [[blog]](https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html)
