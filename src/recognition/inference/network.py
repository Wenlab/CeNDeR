# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import sys

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, Union, List, Optional

from src.common_utils.prints import print_info_message, print_error_message


# ----- Network --------------------
class RecNetworkBase(nn.Module):
    def __init__(self, input_dim: Union[Tuple[int], int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._build_network()

    def _build_network(self):
        raise NotImplementedError()

    def init_weights(self):
        """the weights of conv layer and fully connected layers
        are both initilized with Xavier algorithm, In particular,
        we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)), for batch normalization
        layers, y=1, b=0, all bias initialized to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class RecNetworkMcdropout(RecNetworkBase):

    def __init__(self, input_dim: Union[Tuple[int], int], output_dim: int, dropout_ratio: float):
        super().__init__(input_dim, output_dim)

    def _build_network(self):
        raise NotImplementedError()


# ----- Basic Block --------------------
class NormalizedLinear(nn.Linear):

    def forward(self, input):
        weight_normalize = F.normalize(self.weight, dim = 1)
        return F.linear(input, weight_normalize, self.bias)


class Block1D(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _build_block(self, output_dim, norm_method, activation_method, dropout_ratio, **kwargs):
        layers = list()
        # Normalization Layer
        if norm_method.upper() == "BN":
            layers.append(nn.BatchNorm1d(output_dim))
        elif norm_method.upper() == "GN":
            layers.append(nn.GroupNorm(num_groups = kwargs['num_groups'], num_channels = output_dim))
        elif norm_method.upper() == 'IN':
            layers.append(nn.InstanceNorm1d(num_features = output_dim))

        # Activation Layer
        if activation_method.lower() == "relu":
            layers.append(nn.ReLU(inplace = True))
        elif activation_method.lower() == "relu6":
            layers.append(nn.ReLU6(inplace = True))
        elif activation_method.lower() == "prelu":
            layers.append(nn.PReLU())
        elif activation_method.lower() == "selu":
            layers.append(nn.SELU(inplace = True))
        elif activation_method.lower() == "gelu":
            layers.append(nn.GELU())
        elif activation_method.lower() == "celu":
            layers.append(nn.CELU(inplace = True))

        # Dropout Layer
        if 0 < dropout_ratio < 1.0:
            layers.append(nn.Dropout(dropout_ratio))

        return layers

    def _init_module(self, layers: [nn.Module]):
        module = nn.Sequential(*layers)
        return module

    def forward(self, x):
        x = self.block(x)
        return x


class Linear1DBlock(Block1D):
    def __init__(self, input_dim: int, output_dim: int,
                 norm_method: str = "bn", activation_method: str = 'relu',
                 dropout_ratio: float = 1.0,
                 **kwargs) -> None:
        """
        LNAD (Linear-Normalization-Activation-Dropout)
        :param input_dim:
        :param output_dim:
        :param norm_method:
        :param activation_method:
        :param dropout_ratio:
        :param kwargs:
        """

        super().__init__()
        layers = [nn.Linear(input_dim, output_dim, bias = False)]
        layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        self.block = self._init_module(layers)


class Conv1DBlock(Block1D):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3, stride: int = 1,
                 norm_method: str = "bn", activation_method: str = 'relu',
                 dropout_ratio: float = 1.0,
                 **kwargs) -> None:
        """
        CNAD (Conv-Normalization-Activation-Dropout)
        :param input_dim:
        :param output_dim:
        :param kernel_size:
        :param stride:
        :param norm_method:
        :param activation_method:
        :param dropout_ratio:
        :param kwargs:
        """

        super().__init__()
        layers = [nn.Conv1d(input_dim, output_dim, kernel_size = kernel_size, stride = stride, bias = False)]
        layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        self.block = self._init_module(layers)


# ----- Residual Block --------------------
class Residual1DBlock(Block1D):
    def forward(self, x):
        x = self.line(x) + self.shortcut(x)
        x = self.merge(x)
        return x


class Linear1DResidualBlock(Residual1DBlock):

    def __init__(self, input_dim: int, output_dim: int,
                 norm_method: str = "bn", activation_method: str = 'relu',
                 dropout_ratio: float = 1.0,
                 **kwargs) -> None:
        """

        :param input_dim:
        :param output_dim:
        :param norm_method:
        :param activation_method:
        :param dropout_ratio:
        :param kwargs:
        """

        super().__init__()
        # line block
        line_layers = [nn.Linear(input_dim, output_dim, bias = False)]
        line_layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        line_layers += [nn.Linear(output_dim, output_dim, bias = False)]
        line_layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        line_layers += [nn.Linear(output_dim, output_dim, bias = True)]
        self.line = self._init_module(line_layers)
        # shortcut block
        self.shortcut = nn.Linear(input_dim, output_dim, bias = True)
        # merge block
        self.merge = self._init_module(self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs))


class Conv1DResidualBlock(Residual1DBlock):

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3, stride: int = 1,
                 norm_method: str = "bn", activation_method: str = 'relu',
                 dropout_ratio: float = 1.0,
                 **kwargs) -> None:
        """

        :param input_dim:
        :param output_dim:
        :param kernel_size:
        :param stride:
        :param norm_method:
        :param activation_method:
        :param dropout_ratio:
        :param kwargs:
        """

        super().__init__()
        # line block
        line_layers = [nn.Conv1d(input_dim, output_dim, kernel_size, stride = stride, bias = False)]
        line_layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        line_layers += [nn.Conv1d(output_dim, output_dim, kernel_size, stride = 1, bias = False)]
        line_layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
        line_layers += [nn.Conv1d(output_dim, output_dim, kernel_size, stride = 1, bias = False)]
        self.line = self._init_module(line_layers)
        # shortcut block
        if stride == 1:
            self.shortcut = nn.Conv1d(input_dim, output_dim, kernel_size, stride = 1, bias = True)
        else:
            shortcut_layers = [nn.Conv1d(input_dim, output_dim, kernel_size, stride, bias = False)]
            shortcut_layers += self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs)
            shortcut_layers += [nn.Conv1d(input_dim, output_dim, kernel_size, stride = 1, bias = True)]
            self.shortcut = self._init_module(shortcut_layers)
        # merge block
        self.merge = self._init_module(self._build_block(output_dim, norm_method, activation_method, dropout_ratio, **kwargs))


# ----- Encoding Network --------------------
class RecFuseNetworkLinear(RecNetworkBase):
    def __init__(self, input_dim: Tuple[int, int], output_dim: int, num_ids: int,
                 channel_base: int = 32, group_base: int = 4,
                 dropout_ratio: float = 1.0, activation_method: str = "relu"):
        self.num_ids = num_ids
        self.cb = channel_base
        self.gb = group_base
        self.dratio = dropout_ratio
        self.act = activation_method
        super().__init__(input_dim, output_dim)
        self.init_weights()

    def _build_network(self):
        self.head_knn = Linear1DBlock(self.input_dim[0], self.gb * self.cb, norm_method = "GN", num_groups = self.gb, activation_method = self.act, dropout_ratio = self.dratio)
        self.head_des = Linear1DBlock(self.input_dim[1], self.gb * self.cb, norm_method = 'GN', num_groups = self.gb, activation_method = self.act, dropout_ratio = self.dratio)
        self.head_merge = nn.Sequential(nn.GroupNorm(num_groups = self.gb, num_channels = self.gb * self.cb), nn.ReLU(True))

        self.block1 = Linear1DResidualBlock(self.gb * self.cb, self.gb * self.cb, norm_method = "GN", num_groups = self.gb, activation_method = self.act, dropout_ratio = self.dratio)
        self.block1_merge = nn.Sequential(nn.GroupNorm(num_groups = self.gb, num_channels = self.gb * self.cb), nn.ReLU(True))

        self.block2 = Linear1DResidualBlock(self.gb * self.cb, self.gb * self.cb, norm_method = "GN", num_groups = self.gb, activation_method = self.act, dropout_ratio = self.dratio)
        self.block2_merge = nn.Sequential(nn.GroupNorm(num_groups = self.gb, num_channels = self.gb * self.cb), nn.ReLU(True))

        self.block3 = Linear1DResidualBlock(self.gb * self.cb, self.gb * self.cb, norm_method = "GN", num_groups = self.gb, activation_method = self.act, dropout_ratio = self.dratio)

        self.end = nn.Linear(in_features = self.gb * self.cb, out_features = self.output_dim, bias = True)

        # the distances of samples with every cluster center.
        self.norm_linear = NormalizedLinear(in_features = self.output_dim, out_features = self.num_ids, bias = False)

    def forward(self, feas, mode = 0):
        """

        :param feas:
        :param mode: 0 returns embedding vector; 1 returns 0's result and cos_distance matrix; 2 returns 1's returns and predicted id vector and its probability vector.
        :return:
        """
        local_fea = self.head_knn(feas[:, :self.input_dim[0]])
        global_fea = self.head_des(feas[:, self.input_dim[0]:])
        x = self.block1(self.head_merge(local_fea + global_fea))
        x = self.block2(self.block1_merge(local_fea + x))
        embedding = self.end(self.block3(self.block2_merge(local_fea + x)))

        embedding = F.normalize(embedding, dim = -1)
        if mode == 0:
            results = embedding
        elif mode == 1:
            cos_dis = self.norm_linear(embedding)
            results = (embedding, cos_dis)
        elif mode == 2:
            cos_dis = self.norm_linear(embedding)
            probs = F.softmax(cos_dis, dim = 1)
            results = probs
        elif mode == 3:
            cos_dis = self.norm_linear(embedding)
            probs = F.softmax(cos_dis, dim = 1)
            results = (embedding, probs)
        elif mode == 4:
            cos_dis = self.norm_linear(embedding)
            probs, pred_id = torch.max(F.softmax(cos_dis, dim = 1), dim = 1, keepdim = False)
            results = (embedding, cos_dis, probs, pred_id)
        else:
            print_error_message(f"Mode {mode} doesn't support!")

        return results


# ----- Embedding method --------------------
class RecMarginalCosLossNetwork(nn.Module):
    def __init__(self, len_embedding: int = 32,
                 hypersphere_radius: int = 8,
                 coefficients: Tuple[float] = (1.0, 0.0, 0.0), weights = None) -> None:
        """
            Embedding vector to search clusters for every identity.
            A column vector in the FC matrix is a cluster center vector after training.

        :param len_embedding: the number of dimensions
        :param hypersphere_radius: (also scale) the radius sigma of len_embedding dimension hypersphere
        :param coefficients: (c1, c2, c3) for cos function, cos(c1 * x + c2) - c3, are coefficients of quadratic, linear and consistent term in taylor expansion.
        """

        super().__init__()

        self.cs = coefficients
        self.len_embedding = len_embedding
        self.hypersphere_radius = hypersphere_radius
        self.cross_entropy = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights) if weights is not None else None)

    def forward(self, cos_distance, label):
        onehot_label = torch.zeros_like(cos_distance, dtype = torch.long).scatter(1, label[:, None], 1)
        marginal_cos_distance = torch.cos(self.cs[0] * torch.acos(cos_distance) + self.cs[1]) - self.cs[2]
        modified_cos_matrix = (marginal_cos_distance * onehot_label + cos_distance * (1 - onehot_label)) * self.hypersphere_radius
        loss = self.cross_entropy(modified_cos_matrix, label)
        return loss


# ----- Add Attention Module --------------------
class OreoLayer(nn.Module):
    def __init__(self, num_dim, num_head, num_dim_ffd = 2048, dp_ratio = 0.1, residual_weight = 1.0):
        super().__init__()
        self.res_weight = residual_weight

        self.ffd_0 = nn.Sequential(nn.Linear(num_dim, num_dim_ffd),
                                   nn.ReLU(True),
                                   nn.Dropout(dp_ratio),
                                   nn.Linear(num_dim_ffd, num_dim),
                                   nn.Dropout(dp_ratio))
        self.norm_0 = nn.LayerNorm(num_dim)

        self.self_attn_layer_1 = nn.MultiheadAttention(num_dim, num_head, dropout = dp_ratio)
        self.dropout_1 = nn.Dropout(dp_ratio)
        self.norm_1 = nn.LayerNorm(num_dim)

        self.ffd_2 = nn.Sequential(nn.Linear(num_dim, num_dim_ffd),
                                   nn.ReLU(True),
                                   nn.Dropout(dp_ratio),
                                   nn.Linear(num_dim_ffd, num_dim),
                                   nn.Dropout(dp_ratio))

        self.norm_2 = nn.LayerNorm(num_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # block 0
        residual = x
        x = self.ffd_0(x)
        x = residual * self.res_weight + x
        x = self.norm_0(x)
        # block 1
        residual = x
        x, _ = self.self_attn_layer_1(x, x, x, key_padding_mask = key_padding_mask)
        x = residual * self.res_weight + self.dropout_1(x)
        x = self.norm_1(x)
        # block 2
        residual = x
        x = self.ffd_2(x)
        x = residual * self.res_weight + x
        x = self.norm_2(x)
        return x


class RecEncoder(nn.Module):
    def __init__(self, num_layers, num_dim, num_head, num_dim_ffd = 2048, dp_ratio = 0.1, residual_weight = 1.0):
        super().__init__()

        self.encoder = nn.ModuleList([
            OreoLayer(num_dim = num_dim,
                      num_head = num_head,
                      num_dim_ffd = num_dim_ffd,
                      dp_ratio = dp_ratio,
                      residual_weight = residual_weight)
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(num_dim, num_dim, bias = False)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x, key_padding_mask = src_key_padding_mask)
        embedding = self.final_layer(x)
        return embedding


class RecFormer(nn.Module):
    def __init__(self, input_dim: int, n_hidden: int, n_layer: int = 6, cuda: bool = True, normal: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.cuda = cuda
        self.normal = normal
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self._build_network()
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(mean = 0.0, std = 0.02)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #
    #         elif isinstance(m, nn.MultiheadAttention):
    #             m.q_proj.weight.data.normal_(mean=0.0, std=0.02)
    #             m.k_proj.weight.data.normal_(mean=0.0, std=0.02)
    #             m.v_proj.weight.data.normal_(mean=0.0, std=0.02)

    def _build_network(self):
        self.head = nn.Sequential(nn.Conv1d(self.input_dim, self.n_hidden, 1, bias = False), nn.BatchNorm1d(self.n_hidden), nn.ReLU(True), nn.Dropout(0.1),
                                  nn.Conv1d(self.n_hidden, self.n_hidden, 1), nn.BatchNorm1d(self.n_hidden))
        self.encoder = RecEncoder(num_layers = self.n_layer, num_dim = self.n_hidden, num_head = 8, num_dim_ffd = 2048, dp_ratio = 0.1, residual_weight = 1.0)
        self.fc_outlier = nn.Linear(self.n_hidden, 1)

    def to_input_tensor(self, pts):
        pad_pt = [0, ] * self.input_dim
        sents_padded = []
        max_len = max(len(s) for s in pts)
        for s in pts:
            padded = [pad_pt] * max_len
            padded[:len(s)] = s
            sents_padded.append(padded)
        sents_var = torch.tensor(sents_padded, dtype = torch.float, device = self.device)
        return sents_var

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths) -> torch.Tensor:
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype = torch.float).bool()
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = True
        return enc_masks.to(self.device)

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        model = RecFormer(**args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        params = {
            'args'      : dict(input_dim = self.input_dim,
                               n_hidden = self.n_hidden,
                               n_layer = self.n_layer,
                               cuda = self.cuda,
                               normal = self.normal,
                               ),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def encode(self, pts_padded, pts_length, ref_idx):
        pts_proj = self.head(pts_padded.transpose(2, 1)).transpose(2, 1)
        mask = self.generate_sent_masks(pts_proj, pts_length)
        # append the ref points to each batch
        ref_pts_proj = pts_proj[ref_idx:ref_idx + 1, :pts_length[ref_idx], :]
        mask_ref = torch.zeros((mask.size(0), pts_length[ref_idx]), dtype = torch.float, device = mask.device).bool()

        # simply add 1 to the ref
        ref_pts_proj = torch.repeat_interleave(ref_pts_proj, repeats = pts_proj.size(0), dim = 0) + 2

        pts_proj = torch.cat((ref_pts_proj, pts_proj), dim = 1)
        mask = torch.cat((mask_ref, mask), dim = 1)

        pts_encode = self.encoder(pts_proj.transpose(dim0 = 0, dim1 = 1), src_key_padding_mask = mask)

        return pts_encode.transpose(dim0 = 0, dim1 = 1)

    def forward(self, pts, match_dict = None, ref_idx = 0, mode = 'train'):
        # Compute sentence lengths
        pts_lengths = [len(s) for s in pts]
        # pts_padded should be of dimension (
        pts_padded = self.to_input_tensor(pts)
        pts_encode = self.encode(pts_padded, pts_lengths, ref_idx)

        ref_emb = pts_encode[:, :pts_lengths[ref_idx], :]
        mov_emb = pts_encode[:, pts_lengths[ref_idx]:, :]

        if self.normal:
            ref_emb = F.normalize(ref_emb, dim = -1)
            mov_emb = F.normalize(mov_emb, dim = -1)

        sim_m = torch.bmm(mov_emb, ref_emb.transpose(dim0 = 1, dim1 = 2))
        # the outlier of mov node
        mov_outlier = self.fc_outlier(mov_emb)
        sim_m = torch.cat((sim_m, mov_outlier), dim = 2)

        p_m = F.log_softmax(sim_m, dim = 2)
        p_m_exp = F.softmax(sim_m, dim = 2)

        batch_sz = pts_encode.size(0)
        loss = 0
        num_pt = 0
        loss_entropy = 0
        num_unlabel = 0
        output_pairs = dict()

        if (mode == 'train') or (mode == 'all'):
            for i_w in range(batch_sz):
                # loss for labelled neurons.
                match = match_dict[i_w]
                if len(match) > 0:
                    match_mov = match[:, 0]
                    match_ref = match[:, 1]
                    log_p = p_m[i_w, match_mov, match_ref]
                    loss -= log_p.sum()
                    num_pt += len(match_mov)
                # loss for outliers.
                outlier_list = match_dict['outlier_{}'.format(i_w)]
                if len(outlier_list) > 0:
                    log_p_outlier = p_m[i_w, outlier_list, -1]
                    loss -= log_p_outlier.sum()
                    num_pt += len(outlier_list)

                # Entropy loss for unlabelled neurons.
                unlabel_list = match_dict['unlabel_{}'.format(i_w)]
                if len(unlabel_list) > 0:
                    loss_entropy_cur = p_m[i_w, unlabel_list, :] * p_m_exp[i_w, unlabel_list, :]
                    loss_entropy -= loss_entropy_cur.sum()
                    num_unlabel += len(unlabel_list)

        elif (mode == 'eval') or (mode == 'all'):
            output_pairs['p_m'] = p_m
            paired_idx = torch.argmax(p_m, dim = 1)
            output_pairs['paired_idx'] = paired_idx

        loss_dict = dict()
        loss_dict['loss'] = loss
        loss_dict['num'] = num_pt if num_pt else 1
        loss_dict['loss_entropy'] = loss_entropy
        loss_dict['num_unlabel'] = num_unlabel if num_unlabel else 1

        loss_dict['reg_stn'] = 0
        loss_dict['reg_fstn'] = 0

        return loss_dict, output_pairs


class RecFormer_Test(RecFormer):
    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        model = RecFormer_Test(**args)
        model.load_state_dict(params['state_dict'])
        return model

    def forward(self, pts, pts_lengths):
        pts = self.head(pts.transpose(2, 1)).transpose(2, 1)
        mask = self.generate_sent_masks(pts, pts_lengths)
        pts = self.encoder(pts.transpose(dim0 = 0, dim1 = 1), src_key_padding_mask = mask).transpose(dim0 = 0, dim1 = 1)
        if self.normal:
            pts = F.normalize(pts, dim = -1)
        mov_outlier = self.fc_outlier(pts)
        return pts, mov_outlier
