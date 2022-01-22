# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, Union, List
from munkres import Munkres, make_cost_matrix, Matrix

from common_utils.prints import print_info_message, print_error_message


# ----- Dataset --------------------
class RecFeatureDataset(Dataset):
    def __init__(self,
                 Xs, ys, names,
                 is_train: bool = True,
                 is_fp16: bool = False):
        self.Xs = Xs
        self.ys = ys
        self.names = names
        self.is_train = is_train
        self.is_fp16 = is_fp16

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        X, y = self.Xs[idx], self.ys[idx]
        name = self.names[idx]

        return name, torch.HalfTensor(X) if self.is_fp16 else torch.FloatTensor(X), torch.LongTensor([y])


class InferRecFeatureDataset(Dataset):
    def __init__(self, vols_neurons_feas: Dict,
                 is_fp16: bool = False):
        self.is_fp16 = is_fp16

        self.feas, self.neuron_id = list(), list()
        for vol_name, vol_neurons_feas in vols_neurons_feas.items():
            for neuron_id, fea in vol_neurons_feas.items():
                self.feas.append(fea)
                self.neuron_id.append([vol_name, neuron_id])

    def __len__(self):
        return len(self.neuron_id)

    def __getitem__(self, idx):
        return self.neuron_id[idx], torch.HalfTensor(self.feas[idx]) if self.is_fp16 else torch.FloatTensor(self.feas[idx])


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
    """
    Normalize W dim=1
    """

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
    def __init__(self, input_dim: Union[Tuple[int], int], output_dim: int, num_ids: int,
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

        embedding = F.normalize(embedding, dim = 1)
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


class RecFuseNetworkConv(RecNetworkBase):
    # TODO
    def _build_network(self):
        self.head_knn = nn.Sequential(
                nn.Linear(self.input_dim[0], 3 * 64, bias = False),
                nn.GroupNorm(num_groups = 3, num_channels = 3 * 64),
                nn.PReLU(True),
                nn.Dropout(0.2)
        )


# ----- Embedding method --------------------
class RecMarginalCosLossNetwork(nn.Module):
    def __init__(self, len_embedding: int = 32,
                 hypersphere_radius: int = 8,
                 coefficients: Tuple[float] = (1.0, 0.0, 0.0)) -> None:
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

    def forward(self, cos_distance, label):
        onehot_label = torch.zeros_like(cos_distance, dtype = torch.long).scatter(1, label[:, None], 1)
        marginal_cos_distance = torch.cos(self.cs[0] * torch.acos(cos_distance) + self.cs[1]) - self.cs[2]
        modified_cos_matrix = (marginal_cos_distance * onehot_label + cos_distance * (1 - onehot_label)) * self.hypersphere_radius
        loss = F.cross_entropy(modified_cos_matrix, label)
        return loss


# ----- Infer main procedure --------------------
def nn_infer(args, vols_neurons_feature: Dict, fea_len: Tuple, id_map: Dict):
    # ----------- Dataloader -----------
    dataset = InferRecFeatureDataset(vols_neurons_feature, is_fp16 = args.rec_fp16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = len(id_map), drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)
    # ----------- Network -----------
    network = RecFuseNetworkLinear(input_dim = fea_len, output_dim = args.rec_len_embedding, num_ids = len(id_map), channel_base = args.rec_channel_base,
                                   group_base = args.rec_group_base, dropout_ratio = 0.2, activation_method = "celu").cuda()
    network = network.half() if args.rec_fp16 else network
    network.load_state_dict(torch.load(args.rec_model_load_path, map_location = 'cuda:0')['network'])
    # ----------- Main Procedure -----------
    results = neural_network(dataloader, network, {key: [[], []] for key in vols_neurons_feature.keys()})  # [merged_id, prob]
    return results


def neural_network(dataloader, network, results):
    network.eval()
    neurons_id, prob_matrix = list(), torch.zeros((len(dataloader.dataset), network.num_ids), dtype = torch.float16 if dataloader.dataset.is_fp16 else torch.float32).cuda()
    with torch.no_grad():
        for i, (n_id, fea) in enumerate(tqdm(dataloader, desc = "S.4 recognition.ANN")):
            prob_matrix[i * dataloader.batch_size: i * dataloader.batch_size + len(fea)] = network(fea.cuda(), mode = 2)
            neurons_id.extend([[vol_ctn, int(n)] for vol_ctn, n in zip(*n_id)])

    prob_matrix = np.array(prob_matrix.cpu())
    # # Find the top-k result
    # score_matrix, arg_matrix = torch.sort(preds, dim=1, descending=True)
    # score_matrix, arg_matrix = score_matrix.cpu().numpy(), arg_matrix.cpu().numpy()
    #
    # for (vol_name, n_id), neuron_arg, neuron_score in zip(neurons_id, arg_matrix, score_matrix):
    #     # max_id: [[merged_allocated_id, [id, score], ...]], ...]
    #     pred_vols_neurons_id[vol_name][neuron_arg[0]].append([n_id, [[i, s] for i, s in zip(neuron_arg, neuron_score)]])

    for (vol_name, merged_id), prob_vec in zip(neurons_id, prob_matrix):
        results[vol_name][0].append(merged_id)
        results[vol_name][1].append(prob_vec)

    return {vol_name: [merged_ids, np.array(probs)] for vol_name, (merged_ids, probs) in results.items()}
