'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2022-03-22 16:32:09
LastEditors: Zhang
LastEditTime: 2022-03-23 19:54:56
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from torch import nn, optim

# 下标_i_c 表示粗糙采样, 下标_i_f 表示精细采样

def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gaps, os):
    '''
    粗采样函数 4.(2)
    input:
    os: 射线原点
    ds: 射线方向向量
    N_c: 粗采样点个数
    t_i_c_bin_edges: 粗采样的分段区间左端点
    t_i_c_gaps: 粗采样分段居间长度,在粗采样时为相等值
    '''
    # 采样射线深度t， t服从[t_bin_edges[i]+(i-1)*sample_gap, t_bin_edges[i]+i*sample_gap]之间的均匀分步
    sample_preq = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds)   # [ds.shape[0],ds.shape[1],N_c]
    t_is_c = t_i_c_bin_edges + sample_preq * t_i_c_gaps     
    # 计算由射线原点(os)发射的射线的采样点(r_ts_c)，其采样深度为(t_is_c)，采样方向为(ds)：r(t)=o + t*d
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
    return (r_ts_c, t_is_c)


def get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds):
    '''
    细采样函数 5.2.(5)
    input:
    os: 射线原点
    ds: 射线方向向量
    N_f: 细采样点个数
    t_is_c: 粗采样输出的采样点
    t_f: 最大系采样边界
    w_is_c: 权重,由5.2.(5)定义
    '''

    # 由权重(w_is_c)定义概率密度函数PDFs和概率分步CDFs
    w_is_c = w_is_c + 1e-5  # 防止w_is_c值过小
    pdfs = w_is_c / torch.sum(w_is_c, dim=-1, keepdim=True) # 5.2节,计算分段常数概率密度函数
    cdfs = torch.cumsum(pdfs, dim=-1)
    cdfs = torch.cat([torch.zeros_like(cdfs[..., :1]), cdfs[..., :-1]], dim=-1)

    # 按照以上分别进行均匀采样
    sample_preq = torch.rand(list(cdfs.shape[:-1]) + [N_f]).to(w_is_c)

    idxs = torch.searchsorted(cdfs, sample_preq, right=True) 
    t_i_f_bin_edges = torch.gather(t_is_c, 2, idxs - 1)
    idxs_copped = idxs.clone()
    max_idx = cdfs.shape[-1]
    idxs_copped[idxs_copped==max_idx] = max_idx - 1
    t_i_f_end_edges = torch.gather(t_is_c, 2, idxs_copped)
    t_i_f_end_edges[idxs==max_idx] = t_f
    t_i_f_gaps = t_i_f_end_edges - t_i_f_bin_edges
    sample_preq_f = torch.rand_like(t_i_f_gaps).to(os)
    t_is_f = t_i_f_bin_edges + sample_preq_f*t_i_f_gaps

    # 结合粗采样深度(t_is_c)和细采样深度(t_is_f)，并排序
    (t_is_f, _) = torch.sort(torch.cat([t_is_c, t_is_f.detach()], dim=-1), dim=-1)
    # 计算由射线原点(os)发射的射线的采样点(r_ts_c)，其采样深度为(t_is_f)，采样方向为(ds)：r(t)=o + t*d
    r_ts_f = os[..., None, :] + t_is_f[..., :, None] * ds[..., None, :]
    return (r_ts_f, t_is_f)

def render_radiance_volume(r_ts, ds, chunk_size, F, t_is):
    '''
    渲染函数: 利用神经网络(F)预测射线(r_ts)在方向(ds)上的3D采样点的颜色(c_is)和体积密度(sigma_is).并以此用积分渲染该点.
    input:
    r_ts: 射线的采样点
    ds: 射线的方向向量
    chunk_size: 分块输入神经网络的块大小
    F: 神经网络模型
    t_is: 采样深度
    '''

    r_ts_flat = r_ts.reshape((-1, 3))
    ds_rep = ds.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
    ds_flat = ds_rep.reshape((-1, 3))
    c_is = []
    sigma_is = []
    
    # 按chunk_size分批输入神经网络, 防止显存爆炸
    for chunk_start in range(0, r_ts_flat.shape[0], chunk_size):
        r_ts_batch = r_ts_flat[chunk_start : chunk_start + chunk_size]
        ds_batch = ds_flat[chunk_start : chunk_start + chunk_size]
        preds = F(r_ts_batch, ds_batch)
        c_is.append(preds["c_is"])
        sigma_is.append(preds["sigma_is"])
    
    c_is = torch.cat(c_is).reshape(r_ts.shape)
    sigma_is = torch.cat(sigma_is).reshape(r_ts.shape[:-1])

    # 计算间距(delta_is= t_i+1 - t_i),其中最后一位间距设置为无穷大,可保证4.(3)式中的最后一个alpha始终为1
    delta_is = t_is[..., 1:] - t_is[..., :-1]
    one_e_10 = torch.Tensor([1e10]).expand(delta_is[..., :1].shape)
    delta_is = torch.cat([delta_is, one_e_10.to(delta_is)], dim=-1)
    delta_is = delta_is * ds.norm(dim=-1).unsqueeze(-1)

    # 计算alphas(alpha_is),如式4.(3)
    alpha_is = 1.0 - torch.exp(-sigma_is*delta_is)

    # 计算累积透射率(T_is), 其表示射线从t_n到t_i传播过程中没有与其他粒子碰撞的概率
    T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
    # 滚动tensor-T_is, 使得T_1一定为1,保证射线至少能投射一步
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0

    # 计算权重值(w_is), 帮助后续细采样
    # w_i = T_i * (1 - exp(-sigma_i * delta_i)).
    w_is = T_is * alpha_is

    # 根据射线上采样点颜色(c_is)的加权(w_is)值求得像素点的颜色(C_rs)
    C_rs = (w_is[..., None] * c_is).sum(dim=-2)

    return (C_rs, w_is)

# 定义神经辐射场MLP
class NeRFMLP(nn.Module):
    def __init__(self):
        super(NeRFMLP, self).__init__()
        
        # 位置编码后的特征维度, 坐标位置编码长度为10, 方向位置编码长度为4, 目的是将低维输入转化为高频输入
        self.L_pos = 10
        self.L_dir = 4
        pos_enc_feats = 3 + 3*2*self.L_pos
        dir_enc_feats = 3 + 3*2*self.L_dir

        in_feats = pos_enc_feats   # 输入维度, 注意输入端只输入坐标编码特征, 防止密度预测引入方向信息
        hidden_feats = 256
        
        # 先经过5层的全连接层backbone,提取基本特征信息
        backbone_layers_num = 5
        backbone_layers = []
        for _ in range(backbone_layers_num):
            backbone_layers.append(nn.Linear(in_feats,hidden_feats))
            backbone_layers.append(nn.ReLU())
            in_feats = hidden_feats

        self.backbone_mlp = nn.Sequential(*backbone_layers)

        # 遵循DeepSDF结构,在第5层后引入跳跃连接(concat),进一步提取特征信息
        in_feats = pos_enc_feats + hidden_feats
        concat_layers_num = 3
        concat_layers = []
        for _ in range(concat_layers_num):
            concat_layers.append(nn.Linear(in_feats,hidden_feats))
            concat_layers.append(nn.ReLU())
            in_feats = hidden_feats
        
        self.concat_mlp = nn.Sequential(*concat_layers)
 
        # 输出sigma和特征向量, 注意原文中该层不使用任何激活函数, sigma输出时用ReLU激活函数保证其非负
        self.sigma_layer = nn.Linear(hidden_feats, hidden_feats+1)

        # 预输出层,将方向编码特征融入,并降维特征
        self.pre_final_layer = nn.Sequential(
            nn.Linear(hidden_feats + dir_enc_feats, hidden_feats//2),
            nn.ReLU()
        )

        # 输出RGB颜色颜色值
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_feats//2, 3),
            nn.Sigmoid()
        )

    def forward(self, xs, ds):

        # 坐标特征位置编码,要保留原坐标向量
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2**l_pos*math.pi*xs))
            xs_encoded.append(torch.cos(2**l_pos*math.pi*xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)
        
        # 方向特征位置编码,要保留原方向向量
        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2**l_dir*math.pi*ds))
            ds_encoded.append(torch.cos(2**l_dir*math.pi*ds))
        
        ds_encoded = torch.cat(ds_encoded, dim=-1)

        outputs = self.backbone_mlp(xs_encoded)
        outputs = self.concat_mlp(torch.cat([xs_encoded, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        # 密度输出sigma_is为该层第一维,需经过ReLU激活保证其值非负
        sigma_is = torch.relu(outputs[:,0])
        # sigma层的其余输出特征不经过激活,直接与方向特征编码拼接后输入预输出层
        outputs = self.pre_final_layer(torch.cat([ds_encoded, outputs[:,1:]], dim=-1))
        c_is = self.output_layer(outputs)
        return {"c_is": c_is, "sigma_is": sigma_is}



def run_one_iter_nerf(
    ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, chunk_size, F_c, N_f, t_f, F_f
):
    # 先经过粗采样,获得粗采样点
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    (C_rs_c, w_is_c) = render_radiance_volume(r_ts_c, ds, chunk_size, F_c, t_is_c)
    
    # 根据粗采样点进行细采样
    (r_ts_f, t_is_f) = get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds)
    (C_rs_f, _) = render_radiance_volume(r_ts_f, ds, chunk_size, F_f, t_is_f)

    # 分别输出粗采样渲染和细采样渲染的结果,同时训练两个网络
    return (C_rs_c, C_rs_f)

def main():
    
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 初始化粗细MLPs
    device = "cuda:0"
    F_c = NeRFMLP().to(device)
    F_f = NeRFMLP().to(device)
    # 每次输入神经网络的采样点数量,共4096*64(粗采样)或4096*128(细采样)
    chunk_size = 1024 * 32
    # 输入图像的尺寸
    batch_img_size = 64
    n_batch_pix = batch_img_size**2

    # 初始化优化器
    lr = 5e-4
    optimizer = optim.Adam(list(F_c.parameters()) + list(F_f.parameters()), lr=lr)
    criterion = nn.MSELoss()
    # 学习率进行指数衰减
    lrate_decay = 250
    decay_steps = lrate_decay * 1000
    decay_rate = 0.1

    # 加载数据集
    data_f = "66bdbc812bd0a196e194052f3f12cb2e.npz"
    data = np.load(data_f)

    # 初始化射线原点(init_o)和射线方向(init_ds),设定为相同值后根据相机参数旋转
    images = data["images"]/255
    img_size = images.shape[1]
    xs = torch.arange(img_size) - img_size / 2
    ys = torch.arange(img_size) - img_size / 2
    (xs, ys) = torch.meshgrid(xs, -ys)
    focal = float(data["focal"])
    pixel_coords = torch.stack([xs, ys, -focal * torch.ones_like(xs)], dim=-1)

    camera_coords = pixel_coords / focal
    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

    # 设置测试视角
    test_idx = 150
    plt.imshow(images[test_idx])
    plt.show()
    test_img = torch.Tensor(images[test_idx]).to(device)
    poses = data["poses"]
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    # 初始化体素渲染超参数
    # 射线采样近距离端点
    t_n = 1.0
    # 射线采样远距离端点
    t_f = 4.0
    # 每条射线粗采样点个数
    N_c = 64
    # 每条射线细采样点个数
    N_f = 128
    # 射线分段采样的每段起始位置
    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    # 开始训练模型
    train_idxs = np.arange(len(images)) != test_idx
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])
    n_pix = img_size**2
    pixel_ps = torch.full((n_pix,), 1/n_pix).to(device) # 定义每个像素相同的采样概率
    psnrs = []
    iternums = []

    num_iters = 300000
    display_every = 100
    F_c.train()
    F_f.train()
    for i in range(num_iters):
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        # 根据姿态旋转获得os,ds
        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        os = (R @ init_o).expand(ds.shape)

        # 采样一批射线
        pix_idxs = pixel_ps.multinomial(n_batch_pix, False)  # 把像素展平后采样
        pix_idx_rows = pix_idxs // img_size                  # 把像素的行列还原
        pix_idx_cols = pix_idxs % img_size
        ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )
        os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )

        # Run NeRF.
        (C_rs_c, C_rs_f) = run_one_iter_nerf(
            ds_batch,
            N_c,
            t_i_c_bin_edges,
            t_i_c_gap,
            os_batch,
            chunk_size,
            F_c,
            N_f,
            t_f,
            F_f,
        )
        target_img = images[target_img_idx].to(device)
        target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_f.shape)
        
        # 计算MSELoss
        loss = criterion(C_rs_c, target_img_batch) + criterion(C_rs_f, target_img_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for g in optimizer.param_groups:
            g["lr"] = lr * decay_rate ** (i / decay_steps)

        if i % display_every == 0:
            F_c.eval()
            F_f.eval()
            with torch.no_grad():
                (_, C_rs_f) = run_one_iter_nerf(
                    test_ds,
                    N_c,
                    t_i_c_bin_edges,
                    t_i_c_gap,
                    test_os,
                    chunk_size,
                    F_c,
                    N_f,
                    t_f,
                    F_f,
                )

            loss = criterion(C_rs_f, test_img)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(C_rs_f.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            F_c.train()
            F_f.train()

    print("Done!")


if __name__ == "__main__":
    main()