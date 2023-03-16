import os
import time
import logging
import psutil
import copy
import argparse
import numpy as np
import cv2 as cv
import open3d as o3d
import trimesh
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from skimage.measure import marching_cubes
from models.fields import NeRF
from models.my_dataset import Dataset
from models.my_nerf import MyNeRF, CheatNeRF
from models.my_renderer import MyNerfRenderer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings

warnings.filterwarnings('ignore')

RS = 512
class Runner:
    def __init__(self, conf_path, ds='voxel', mode='render', case='CASE_NAME', is_continue=False, checkpoint=None):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda:0')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'], self.device)
        self.iter_step = 0
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.coarse_nerf = NeRF(**self.conf['model.coarse_nerf']).to(self.device)
        self.fine_nerf = NeRF(**self.conf['model.fine_nerf']).to(self.device)
        if is_continue and checkpoint is not None:
            self.my_nerf = MyNeRF(RS, ds, self.device, checkpoint)
        else:
            self.my_nerf = MyNeRF(RS, ds, self.device)
        self.renderer = MyNerfRenderer(self.my_nerf,
                                     **self.conf['model.nerf_renderer'])
        self.load_checkpoint(r'E:\Document\sci_research\beginning\homework\code\nerf_model.pth', absolute=True)


    def load_checkpoint(self, checkpoint_name, absolute=False):
        if absolute:
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.coarse_nerf.load_state_dict(checkpoint['coarse_nerf'])
        self.fine_nerf.load_state_dict(checkpoint['fine_nerf'])
        logging.info('End')

    def use_nerf(self):
        self.my_nerf = CheatNeRF(self.fine_nerf)
        self.renderer = MyNerfRenderer(self.my_nerf,
                                     **self.conf['model.nerf_renderer'])

    def save_volume(self):
        pts_xyz = torch.zeros((RS, RS, RS, 3), device=self.device, requires_grad=False)
        for i in tqdm(range(RS)):
            for j in range(RS):
                pts_xyz[:, i, j, 0] = torch.linspace(-0.125, 0.125, RS).to(self.device)
                pts_xyz[i, :, j, 1] = torch.linspace(0.75, 1.0, RS).to(self.device)
                pts_xyz[i, j, :, 2] = torch.linspace(-0.125, 0.125, RS).to(self.device)
        pts_xyz = pts_xyz.reshape((RS*RS*RS, 3))
        # checkpoint = {
        #     "pts_xyz": pts_xyz
        # }
        # file = 'pts_xyz_' + str(RS) + '.pth'
        # torch.save(checkpoint, os.path.join('checkpoints', file))
        batch_size = 2048
        volume_sigma = torch.zeros((RS*RS*RS, 1), device=self.device, requires_grad=False)
        volume_color = torch.zeros((RS*RS*RS, 3), device=self.device, requires_grad=False)
        # volume_sigma = torch.zeros((RS, RS, RS, 1), device=self.device, requires_grad=False)
        # volume_color = torch.zeros((RS, RS, RS, 3), device=self.device, requires_grad=False)
        for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
            batch_pts_xyz = pts_xyz[batch:batch+batch_size]
            # Z_index = ((batch_pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
            # X_index = ((batch_pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
            # Y_index = ((batch_pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
            net_sigma, net_color = self.fine_nerf(batch_pts_xyz, torch.zeros_like(batch_pts_xyz, device=self.device))
            volume_sigma[batch:batch+batch_size] = net_sigma.detach()
            volume_color[batch:batch+batch_size] = net_color.detach()
            # volume_sigma[X_index, Y_index, Z_index] = net_sigma.detach()
            # volume_color[X_index, Y_index, Z_index] = net_color.detach()
        checkpoint = {
            "volume_sigma": volume_sigma,
            "volume_color": volume_color
        }
        file = 'voxelview_' + str(RS) + '.pth'
        torch.save(checkpoint, os.path.join('checkpoints', file))

    def save(self):
        pts_xyz = torch.zeros((RS, RS, RS, 3), device=self.device, requires_grad=False)
        for i in tqdm(range(RS)):
            for j in range(RS):
                pts_xyz[:, i, j, 0] = torch.linspace(-0.125, 0.125, RS).to(self.device)
                pts_xyz[i, :, j, 1] = torch.linspace(0.75, 1.0, RS).to(self.device)
                pts_xyz[i, j, :, 2] = torch.linspace(-0.125, 0.125, RS).to(self.device)
        pts_xyz = pts_xyz.reshape((RS * RS * RS, 3))

        if self.my_nerf.ds == 'voxel':
            batch_size = 2048
            volume_sigma = torch.zeros((RS*RS*RS, 1), device=self.device, requires_grad=False)
            volume_color = torch.zeros((RS*RS*RS, 3), device=self.device, requires_grad=False)
            for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
                batch_pts_xyz = pts_xyz[batch:batch + batch_size]
                net_sigma, net_color = self.fine_nerf(batch_pts_xyz,
                                                      torch.zeros_like(batch_pts_xyz, device=self.device))
                volume_sigma[batch:batch+batch_size] = net_sigma.detach()
                volume_color[batch:batch+batch_size] = net_color.detach()
            self.my_nerf.save(pts_xyz, volume_sigma, volume_color)
        elif self.my_nerf.ds == 'octree':
            self.my_nerf.save(pts_xyz, None, None)

    def render_video(self):
        images = []
        resolution_level = 1
        n_frames = 90
        for idx in tqdm(range(n_frames)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(1024)
            rays_d = rays_d.reshape(-1, 3).split(1024)

            out_rgb_fine = []

            for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3], device=self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('fine_color'):
                    out_rgb_fine.append(render_out['fine_color'].detach().cpu().numpy())

                del render_out
            
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)
            os.makedirs(os.path.join(self.base_exp_dir, 'render_'+str(RS)+self.my_nerf.ds), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir,  'render_'+str(RS)+self.my_nerf.ds, '{}.jpg'.format(idx)), img_fine)

        # fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # h, w, _ = images[0].shape
        # writer = cv.VideoWriter(os.path.join(self.base_exp_dir,  'render', 'show.mp4'),
        #                         fourcc, 30, (w, h))
        # for image in tqdm(images):
        #     writer.write(image)
        # writer.release()

    def extract_mesh(self):
        volume_sigma = torch.load('checkpoints/voxel_'+str(RS)+'.pth')['volume_sigma'].view(RS, RS, RS)
        max_sigma = float(torch.max(volume_sigma))
        min_sigma = float(torch.min(volume_sigma))
        # # ostu
        # sigma_temp = torch.histc(self.my_nerf.volume_sigma.reshape(-1), 100, min_sigma, max_sigma)
        # p = sigma_temp / sigma_temp.sum()
        # u = (torch.arange(0, 100).to(self.device) * p).sum()
        # max_index = 0
        # max_var = 0
        # for i in range(1, 100):
        #     omega_0 = p[:i].sum()
        #     omega_1 = 1 - omega_0
        #     u_0 = (torch.arange(0, i).to(self.device) * p[:i]).sum() / omega_0
        #     u_1 = (u - (torch.arange(0, i).to(self.device) * p[:i]).sum()) / omega_1
        #     var = omega_0 * (u_0 - u)**2 + omega_1 * (u_1 - u)**2
        #     print(omega_0, var)
        #     if var > max_var:
        #         max_index = i
        #         max_var = var
        # threshold = min_sigma + (max_sigma - min_sigma) * max_index / 100
        # print(max_index, threshold)
        threshold = min_sigma + (max_sigma - min_sigma) * 0.62
        # threshold = min_sigma + (max_sigma - min_sigma) * 0.66
        sigma = volume_sigma > threshold
        # checkpoint = {
        #     "voxel01": sigma
        # }
        # file = 'voxel01_' + str(RS) + '.pth'
        # torch.save(checkpoint, os.path.join('checkpoints', file))
        # print(sigma.sum())
        #
        vertices, faces, _, _ = marching_cubes(sigma.cpu().numpy())
        trimesh.Trimesh(vertices, faces).export(os.path.join('objs', 'test'+str(RS)+'.obj'))
        # trimesh.Trimesh(vertices, faces)

    def get_voxel01(self):
        rs=128
        mesh = o3d.io.read_triangle_mesh('objs/test'+str(rs)+'.obj')
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                   center=mesh.get_center())
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1/rs)
        print(voxel_grid)

if __name__ == '__main__':

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--ds', type=str, default='voxel')
    parser.add_argument('--mode', type=str, default='render')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='')

    args = parser.parse_args()
    runner = Runner(args.conf, args.ds, args.mode, args.case, args.is_continue, args.checkpoint)

    if args.mode == 'render':
        # runner.save()
        runner.render_video()
    elif args.mode == 'test':
        runner.use_nerf()
        runner.render_video()
    elif args.mode == 'mesh':
        runner.extract_mesh()
    elif args.mode == 'save':
        runner.save_volume()
    elif args.mode == 'voxel':
        runner.get_voxel01()
    elif args.mode == 'none':
        pass