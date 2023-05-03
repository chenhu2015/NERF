import os
import sys
import torch
import torch.nn as nn
import numpy as np
import imageio
import json
from typing import Optional, Tuple
import torch.nn.functional as F


# Misc utils

def img2mse(x, y): return torch.mean(torch.square(x - y))


def mse2psnr(x): return -10.*torch.log(x)/torch.log(torch.Tensor([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


'''
# Model architecture
def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

'''


# Model architecture
class NeRF(nn.Module):
  r"""
  Neural radiance fields module.
  """
  def __init__(
    self,
    d_input: int = 3,
    d_input_views: int = 0,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: list = [4],
    d_viewdirs: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.d_input_views = d_input_views
    self.skip = skip
    self.act = nn.functional.relu
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    self.branch = nn.Linear(d_filter + d_input_views, d_filter // 2)

    # Bottleneck layers
    if self.d_viewdirs:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor
    ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.
    """
    # Apply forward pass up to bottleneck
    x_input, x_views = torch.split(x, [self.d_input, self.d_input_views], dim = -1)
    x = x_input

    for i, layer in enumerate(self.layers):
      x = self.layers[i](x)
      x = F.relu(x, inplace=False)

      if i in self.skip:
        x = torch.cat([x_input, x], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs:
      # Split alpha from network output
      alpha = self.alpha_out(x)
      feature = self.rgb_filters(x)
      x = torch.cat([feature, x_views], dim=-1)

      x = self.branch(x)
      x = F.relu(x, inplace=False)
      rgb = self.output(x)

      # Concatenate alphas to output
      res = torch.cat([rgb, alpha], dim=-1)
    else:
      # Simple output
      res = self.output(x)

    return res


# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    focal = np.array([[focal, 0, W * 0.5], [0, focal, H * 0.5], [0, 0, 1]])

    i, j = torch.meshgrid(torch.linspace(0, W-1, W),
                       torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-focal[0][2])/focal[0][0], -(j-focal[1][2])/focal[1][1], -torch.ones_like(i)], dim = -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim = -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    focal = np.array([[focal, 0, W * 0.5], [0, focal, H * 0.5], [0, 0, 1]])

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-focal[0][2])/focal[0][0], -(j-focal[1][2])/focal[1][1], -np.ones_like(i)], dim = -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim = -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], dim = -1)
    rays_d = torch.stack([d0, d1, d2], dim = -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False, test=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, dim = -1, keepdims=True)
    cdf = torch.cumsum(pdf, dim = -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim = -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    if test:
      np.random.seed(0)
      new_s = list(cdf.shape[:-1]) + [N_samples]

      if det:
        u = np.linspace(0., 1., N_samples)
        u = np.broadcast_to(u, new_s)
      else:
        u = np.random.rand(*new_s)

      u = torch.Tensor(u)

    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim = -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(torch.broadcast_to(cdf.unsqueeze(1), matched_shape), dim = 2, index = inds_g)
    bins_g = torch.gather(torch.broadcast_to(bins.unsqueeze(1), matched_shape), dim = 2, index = inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples