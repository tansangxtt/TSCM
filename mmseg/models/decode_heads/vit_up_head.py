import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
# import torch.distributed as dist

# from .layers import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@HEADS.register_module()
class VisionTransformerUpHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, embed_dim=1024,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                 num_conv=1, upsampling_method='bilinear', num_upsampe_layer=1, conv3x3_conv1x1=True, **kwargs):
        super(VisionTransformerUpHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv
        self.norm = norm_layer(embed_dim)
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer
        self.conv3x3_conv1x1 = conv3x3_conv1x1
        out_channel = self.num_classes

        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Conv2d(
                    embed_dim, 256, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_0 = nn.Conv2d(embed_dim, 256, 1, 1)
            self.conv_1 = nn.Conv2d(256, out_channel, 1, 1)
            _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)

        elif self.num_conv == 4:
            self.conv_0 = nn.Conv2d(
                embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_4 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1)

            _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)

            # dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
        # Segmentation head

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        print("x", x[0].shape, len(x))
        x = self._transform_inputs(x)
        print("_transform_inputs(x)", x.shape)
        if x.dim() == 3:
            if x.shape[1] % 48 != 0:
                x = x[:, 1:]
            x = self.norm(x)

        if self.upsampling_method == 'bilinear':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))

                # set hard code
                h = 30
                w = 53
                x = x.transpose(1, 2).reshape(n, c, h, w)

            if self.num_conv == 2:
                print("num_conv", self.num_conv)
                if self.num_upsampe_layer == 2:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
                elif self.num_upsampe_layer == 1:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_conv == 4:
                if self.num_upsampe_layer == 4:
                    x = self.conv_0(x)
                    print(x.shape)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = self.syncbn_fc_1(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_2(x)
                    x = self.syncbn_fc_2(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_3(x)
                    x = self.syncbn_fc_3(x)
                    x = F.relu(x, inplace=True)
                    x = self.conv_4(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        return x
