import torch
import svox
import numpy as np
from tqdm import tqdm
import time


class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf

    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))


class MyNeRF():

    def __init__(self, RS, ds, device, **kwargs):
        super(MyNeRF, self).__init__()
        self.RS = RS
        self.ds = ds
        self.device = device
        if ds == 'voxel':
            if 'checkpoint' in kwargs:
                checkpoint = torch.load(kwargs['checkpoint'])
                self.volume_sigma = checkpoint['volume_sigma']
                self.volume_color = checkpoint['volume_color']
        elif ds == 'octree':
            self.octree = svox.N3Tree(center=[0., 0.875, 0.], radius=0.125, init_refine=6, device=device)
            checkpoint = torch.load('checkpoints/voxel_128.pth')
            self.volume_sigma = checkpoint['volume_sigma'].to(device)
            self.volume_color = checkpoint['volume_color'].to(device)
            self.save()

    def save(self):

        if self.ds == 'voxel':
            pass

        elif self.ds == 'octree':
            print("Saving to octree...")
            # 将128voxel赋给tree
            pts_xyz128 = torch.load('checkpoints/pts_xyz_128.pth')['pts_xyz'].to(self.device).detach()
            self.octree[pts_xyz128] = torch.hstack((self.volume_color.view(-1, 3), self.volume_sigma.view(-1, 1)))
            # print(self.octree[pts_xyz128][:,:3].sum())
            # print(torch.max(self.octree[pts_xyz128][:,:3]))
            # 对表面附近的区域进行细分
            depth = 6
            while depth <= 7:
                # 算出每个node对应的区域范围
                indexes_to_expand = (self.octree.depths == depth).to(self.device)
                # print(type(indexes_to_expand))
                corners = self.octree[indexes_to_expand].corners  # (n_nodes, 3)
                # corners_end = corners_begin + 2 ** (-depth - 1) * 0.25
                X_index, Y_index, Z_index = abs_to_rel(corners, 2 ** (depth + 1))
                # corners_end = (abs_to_rel(corners_end, 128) + 5).clamp(0, 127).long()
                # print('depth', depth, corners_begin)
                # 将范围中包含边界的那些node的索引求出来
                voxel01 = torch.load('checkpoints/voxel01_' + str(2 ** (depth + 1)) + '.pth')['voxel01'].to(self.device)
                expand_voxel01(voxel01, depth-5)
                indexes = voxel01[X_index, Y_index, Z_index]
                # indexes = [self.voxel01[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]].any()
                #            for begin, end in torch.stack((corners_begin, corners_end)).permute(1, 0, 2)]
                # print(indexes)
                # indexes = torch.Tensor(indexes)
                # 继续细分
                indexes_to_expand[indexes_to_expand.clone()] = indexes
                self.octree[indexes_to_expand].refine()
                depth += 1
            # 将边界附近的对应的512voxel进行赋值
            pts_xyz512 = self.octree[self.octree.depths == 8].corners
            X_index, Y_index, Z_index = abs_to_rel(pts_xyz512, 512)
            checkpoint = torch.load('checkpoints/voxel_512.pth')
            sigma = checkpoint['volume_sigma'].to(self.device)
            color = checkpoint['volume_color'].to(self.device)
            self.octree[pts_xyz512] = torch.hstack(
                (color[X_index, Y_index, Z_index].view(-1, 3), sigma[X_index, Y_index, Z_index].view(-1, 1)))
            self.voxel01 = torch.load('checkpoints/voxel01_512.pth')['voxel01'].to(self.device)
            expand_voxel01(self.voxel01, 3)
            print("Octree is saved!")

    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)

        if self.ds == 'voxel':
            X_index = ((pts_xyz[:, 0] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()

            sigma[:, 0] = self.volume_sigma[X_index, Y_index, Z_index]
            color[:, :] = self.volume_color[X_index, Y_index, Z_index]

        elif self.ds == 'octree':
            # 表面附近的点
            # X_index, Y_index, Z_index = abs_to_rel(pts_xyz, 512)
            features = self.octree[pts_xyz]
            sigma = features[:, 3:]
            color = features[:, :3]
            # indexes = self.voxel01[X_index, Y_index, Z_index]
            # # indexes = torch.zeros(N, dtype=bool).to(self.device)
            # if len(indexes) > 0:
            #     pts_xyz_surface = pts_xyz[indexes]
            #     # print(type(pts_xyz_surface), pts_xyz_surface.shape)
            #     features = self.octree[rel_to_abs(pts_xyz_surface, 512)]
            #     sigma[indexes] = features[:, 3:]
            #     color[indexes] = features[:, :3]
            # # 其余点
            # pts_xyz_others = pts_xyz[~indexes]
            # X_index = ((pts_xyz_others[:, 0] + 0.125) * 4 * 128).clamp(0, 127).long()
            # Y_index = ((pts_xyz_others[:, 1] - 0.75) * 4 * 128).clamp(0, 127).long()
            # Z_index = ((pts_xyz_others[:, 2] + 0.125) * 4 * 128).clamp(0, 127).long()
            # sigma[~indexes] = self.volume_sigma[X_index, Y_index, Z_index]
            # color[~indexes] = self.volume_color[X_index, Y_index, Z_index]
        # print(torch.max(self.octree[pts_xyz][:, :3]))
        # print(torch.max(color))
        return sigma, color


def abs_to_rel(abs_pts_xyz, RS=128):
    # rel_pts_xyz = torch.zeros_like(abs_pts_xyz)
    X_index = ((abs_pts_xyz[..., 0] + 0.125) * 4 * RS).clamp(0, RS - 1).clone().long()
    Y_index = ((abs_pts_xyz[..., 1] - 0.75) * 4 * RS).clamp(0, RS - 1).clone().long()
    Z_index = ((abs_pts_xyz[..., 2] + 0.125) * 4 * RS).clamp(0, RS - 1).clone().long()
    return X_index, Y_index, Z_index


def rel_to_abs(rel_pts_xyz, RS=128):
    abs_pts_xyz = torch.zeros_like(rel_pts_xyz)
    abs_pts_xyz[..., 0] = (rel_pts_xyz[..., 0] / RS / 4 - 0.125).clone()
    abs_pts_xyz[..., 1] = (rel_pts_xyz[..., 1] / RS / 4 + 0.75).clone()
    abs_pts_xyz[..., 2] = (rel_pts_xyz[..., 2] / RS / 4 - 0.125).clone()
    return abs_pts_xyz


def expand_voxel01(voxel01, expand_size):
    # expanded = voxel01.clone()
    for i in range(1, expand_size + 1):
        voxel01[: - i, :, :] = torch.logical_or( voxel01[: - i, :, :], voxel01[i:, :, :])
    for i in range(1, expand_size + 1):
        voxel01[i:, :, :] = torch.logical_or( voxel01[i:, :, :], voxel01[: - i, :, :])
    for i in range(1, expand_size + 1):
        voxel01[:, : - i, :] = torch.logical_or( voxel01[:, : - i, :], voxel01[:, i:, :])
    for i in range(1, expand_size + 1):
        voxel01[:, i:, :] = torch.logical_or( voxel01[:, i:, :], voxel01[:, : - i, :])
    for i in range(1, expand_size + 1):
        voxel01[:, :, : - i] = torch.logical_or( voxel01[:, :, : - i], voxel01[:, :, i:])
    for i in range(1, expand_size + 1):
        voxel01[:, :, i:] = torch.logical_or( voxel01[:, :, i:], voxel01[:, :, : - i])
