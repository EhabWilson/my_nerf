import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, RS, ds, device, checkpoint=None):
        super(MyNeRF, self).__init__()
        self.RS = RS
        self.ds = ds
        self.device = device
        if ds == 'voxel':
            if checkpoint is not None:
                checkpoint = torch.load(checkpoint)
                self.volume_sigma = checkpoint["volume_sigma"]
                self.volume_color = checkpoint["volume_color"]
            else:
                self.volume_sigma = torch.zeros((RS, RS, RS), device=device)
                self.volume_color = torch.zeros((RS, RS, RS, 3), device=device)
        elif ds == 'tensoRF':
            self.tensoRF = tensoRF(RS, device)

    def save(self, pts_xyz, sigma, color):

        if self.ds == 'voxel':
            X_index = ((pts_xyz[:, 0] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()
            self.volume_sigma[X_index, Y_index, Z_index] = sigma[:, 0]
            self.volume_color[X_index, Y_index, Z_index] = color[:, :]

        elif self.ds == 'tensoRF':
            print(torch.max(sigma), torch.min(sigma), torch.mean(sigma), torch.var(sigma))
            print(torch.max(color), torch.min(color), torch.mean(color))
            lr = 1e-1
            batch_size = 2048
            epochs = 30
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(self.tensoRF.parameters(),
                                        lr=lr,
                                        weight_decay=0.0001,
                                        momentum=0.9)
            writer = SummaryWriter(comment="lr" + str(lr) + "_batchsize" + str(batch_size))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
            # reg = TVLoss()
            for epoch in tqdm(range(epochs)):
                sigma_loss = 0.
                app_loss = 0.
                total_loss = 0.

                # warmup
                if epoch == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-2
                elif epoch == 5:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-1

                for batch in range(0, pts_xyz.shape[0], batch_size):
                    batch_pts_xyz = pts_xyz[batch:batch + batch_size]
                    sigma_pred, color_pred = self.tensoRF(batch_pts_xyz)
                    # print(torch.max(sigma_pred), torch.min(sigma_pred))
                    # print(torch.max(sigma[batch:batch+batch_size]), torch.min(sigma[batch:batch+batch_size]))
                    # loss = torch.mean((sigma_pred - sigma[batch:batch+batch_size].view(-1)) ** 2) \
                    #         + 1e-4*self.tensoRF.density_L1() + 0.1*self.tensoRF.TV_loss_density(reg) \
                    #         + 1e-2*self.tensoRF.TV_loss_app(reg)
                    loss = criterion(sigma_pred, sigma[batch:batch + batch_size].view(
                        -1))# * 1e-4 + 0.16)  # + 1e-6 * self.tensoRF.density_norm()
                    # loss = loss + criterion(color_pred, color[batch:batch + batch_size] - 0.0391)
                    writer.add_scalar('iter_loss', loss, (epoch * pts_xyz.shape[0] + batch) / batch_size)
                    total_loss += loss.detach().item()
                    optimizer.zero_grad()
                    loss.backward()

                    # loss = torch.sum((color_pred - color[batch:batch+batch_size]) ** 2) + self.tensoRF.TV_loss_density(1e-3)
                    # # writer.add_scalar('app_loss', loss, (epoch * pts_xyz.shape[0] + batch) / batch_size)
                    # app_loss += loss.item()
                    # loss.backward()

                # writer.add_scalar('sigma_loss', sigma_loss, epoch)
                # writer.add_scalar('app_loss', app_loss, epoch)
                writer.add_scalar('total_loss', total_loss, epoch)
                print('loss:', total_loss, torch.max(self.tensoRF.density_plane[0]))
                optimizer.step()
                scheduler.step()

    def query(self, pts_xyz):
        if self.ds == 'voxel':
            N, _ = pts_xyz.shape
            sigma = torch.zeros(N, 1, device=pts_xyz.device)
            color = torch.zeros(N, 3, device=pts_xyz.device)
            X_index = ((pts_xyz[:, 0] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * self.RS).clamp(0, self.RS - 1).long()
            Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long()

            sigma[:, 0] = self.volume_sigma[X_index, Y_index, Z_index]
            color[:, :] = self.volume_color[X_index, Y_index, Z_index]

        elif self.ds == 'tensoRF':
            with torch.no_grad():
                # plane, line = self.tensoRF.get_coordinate(pts_xyz)
                sigma, color = self.tensoRF(pts_xyz)

        return sigma, color + 0.0391# * 1e4 + 1600, color + 0.0391


class tensoRF(nn.Module):
    def __init__(self, RS, device):
        super(tensoRF, self).__init__()

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.gridSize = [RS, RS, RS]
        self.app_n_comp = 16
        self.density_n_comp = 8
        self.app_dim = 3
        self.device = device
        self.RS = RS

        self.density_plane, self.density_line = self.init_one_svd([8, 8, 8], self.gridSize, 0.15, device)
        self.app_plane, self.app_line = self.init_one_svd([16, 16, 16], self.gridSize, 0.5, device)
        self.basis_mat = torch.nn.Linear(48, 3, bias=False).to(device)

    def init_one_svd(self, channels, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(3):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            # my_tensor
            # plane_coef.append(torch.nn.Parameter(
            #     scale * torch.randn((channels[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            # line_coef.append(
            #     torch.nn.Parameter(scale * torch.randn((channels[i], gridSize[vec_id], 1, 1))))
            # official
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, channels[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, channels[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # def forward(self, pts_xyz):
    #     N = pts_xyz.shape[0]
    #     X_index = ((pts_xyz[:, 0] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long().detach()
    #     Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * self.RS).clamp(0, self.RS - 1).long().detach()
    #     Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * self.RS).clamp(0, self.RS - 1).long().detach()
    #
    #     sigma = torch.zeros(N, 1, device=pts_xyz.device, requires_grad=True)
    #     color = torch.zeros(N, 3, device=pts_xyz.device, requires_grad=True)
    #
    #     for idx_plane in range(len(self.density_plane)):
    #         # [8, 512, 512]
    #         # [8, 512, 1]
    #         for n_channel in range(len(self.density_plane[idx_plane])):
    #             temp = self.density_line[idx_plane][n_channel] * self.density_plane[idx_plane][n_channel]
    #             temp = temp.permute(self.matMode[idx_plane][0], self.matMode[idx_plane][1], self.vecMode[idx_plane])
    #             with torch.no_grad():
    #                 temp = temp[X_index, Y_index, Z_index].view(-1, 1)
    #             sigma = sigma + temp
    #
    #     return sigma, color

    def forward(self, origin_pts_xyz):
        pts_xyz = origin_pts_xyz.clone().detach()
        pts_xyz[:, 0] = (pts_xyz[:, 0] + 0.125) / 0.25 * 2 - 1
        pts_xyz[:, 1] = (pts_xyz[:, 1] - 0.75) / 0.25 * 2 - 1
        pts_xyz[:, 2] = (pts_xyz[:, 2] + 0.125) / 0.25 * 2 - 1

        coordinate_plane = torch.stack((pts_xyz[..., self.matMode[0]], pts_xyz[..., self.matMode[1]],
                                        pts_xyz[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (pts_xyz[..., self.vecMode[0]], pts_xyz[..., self.vecMode[1]], pts_xyz[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1,
                                                                                                                  1, 2)

        sigma_feature = torch.zeros((pts_xyz.shape[0],), device=self.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                             align_corners=True).view(-1, *pts_xyz.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *pts_xyz.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *pts_xyz.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *pts_xyz.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        app_features = self.basis_mat((plane_coef_point * line_coef_point).T)

        return sigma_feature, app_features

    def density_norm(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.norm(self.density_plane[idx]) ** 2 + torch.norm(self.density_line[idx]) ** 2
        return total

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[
                                                                                                      idx]))  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]