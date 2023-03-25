import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase
import pdb


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]

        n_rays = ray_bundle.origins.shape[0]
        # z_vals = ((self.min_depth - self.max_depth) * torch.rand(size = (n_rays,self.n_pts_per_ray) ) + self.max_depth).cuda() #TODO: friggin doubt about scaling distributions
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).unsqueeze(-1).cuda()
        # z_vals = z_vals.unsqueeze(-1).unsqueeze(0).repeat(n_rays,1,1)

        # TODO (1.4): Sample points from z values
        # ray_bundle.directions = ray_bundle.directions.unsqueeze(1).cuda()
        # pdb.set_trace()
        # sample_points = (z_vals*ray_bundle.directions + ray_bundle.origins.unsqueeze(1).repeat(1,z_vals.shape[1],1)).cuda()
        sample_points = ray_bundle.origins.unsqueeze(1) + z_vals * ray_bundle.directions.unsqueeze(1)


        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}