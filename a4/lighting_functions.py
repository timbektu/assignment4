import numpy as np
import torch
from ray_utils import RayBundle

def phong(
    normals,
    view_dirs, 
    light_dir,
    params,
    colors
):
    # TODO: Implement a simplified version Phong shading
    # Inputs:
    #   normals: (N x d, 3) tensor of surface normals
    #   view_dirs: (N x d, 3) tensor of view directions
    #   light_dir: (3,) tensor of light direction
    #   params: dict of Phong parameters
    #   colors: (N x d, 3) tensor of colors
    # Outputs:
    #   illumination: (N x d, 3) tensor of shaded colors
    #
    # Note: You can use torch.clamp to clamp the dot products to [0, 1]
    # Assume the ambient light (i_a) is of unit intensity 
    # While the general Phong model allows rerendering with multiple lights, 
    # here we only implement a single directional light source of unit intensity

    ka = params['ka']
    kd = params['kd']
    ks = params['ks']
    alpha = params['n']

    n_l = torch.clamp(torch.sum(light_dir*normals, dim=1), 0, 1)
    r_dirs = torch.clamp(
        2 * torch.sum(normals * light_dir.view(1, -1, 3),dim=2, keepdim=True) * normals - light_dir.view(1, -1, 3), 
        0, 1)
    r_v = torch.sum(r_dirs * view_dirs.view(1, -1, 3), dim=2, keepdim=True)
    r_v = torch.clamp(r_v, min=0, max=1)

    ambi = ka*colors
    diffuse = kd* n_l.view(-1,1) *colors
    specular = ks * (r_v.view(-1,1)**alpha) * colors

    return ambi + diffuse + specular





relighting_dict = {
    'phong': phong
}
