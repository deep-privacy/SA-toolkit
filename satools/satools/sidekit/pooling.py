"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher
"""

import torch
import math

class MeanStdPooling(torch.nn.Module):
    """
    Mean and Standard deviation pooling
    """
    def __init__(self):
        """

        """
        super(MeanStdPooling, self).__init__()
        pass

    def forward(self, x):
        """

        :param x: [B, C*F, T]
        :return:
        """
        if len(x.shape) == 4:
            # [B, C, F, T]
            x = x.permute(0, 1, 3, 2)
            x = x.flatten(start_dim=1, end_dim=2)
        # [B, C*F]
        mean = torch.mean(x, dim=2)
        # [B, C*F]
        std = torch.std(x, dim=2)
        # [B, 2*C*F]
        return torch.cat([mean, std], dim=1)


class ChannelWiseCorrPooling(torch.nn.Module):
    """

    """
    def __init__(self, in_channels=256, out_channels=64, in_freqs=10, channels_dropout=0.25):
        super(ChannelWiseCorrPooling, self).__init__()
        self.channels_dropout = channels_dropout
        self.merge_freqs_count = 2
        assert in_freqs % self.merge_freqs_count == 0
        self.groups = in_freqs//self.merge_freqs_count
        self.out_channels = out_channels
        self.out_dim = int(self.out_channels*(self.out_channels-1)/2)*self.groups
        self.L_proj = torch.nn.Conv2d(in_channels*self.groups, out_channels*self.groups, kernel_size=(1, 1), groups=self.groups)
        # self.L_proj = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.mask = torch.tril(torch.ones((out_channels, out_channels)), diagonal=-1).type(torch.BoolTensor)

    def forward(self, x):
        """

        :param x: [B, C, T, F]
        :return:
        """
        batch_size=x.shape[0]
        num_locations = x.shape[-1]*x.shape[-2]/self.groups
        self.mask = self.mask.to(x.device)
        if self.training:
            x *= torch.nn.functional.dropout(torch.ones((1, x.shape[1], 1, 1), device=x.device), p=self.channels_dropout)
        # [B, T, C, F]
        x = x.permute(0, 2, 1, 3)
        # [B, T, C, Fr, f]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[-2], self.groups, self.merge_freqs_count)
        # [B, T, f, Fr, C]
        x = x.permute(0, 1, 4, 3, 2)
        # [B, T, f, Fr*C]
        x = x.flatten(start_dim=3, end_dim=4)
        # [B, Fr*C, T, f]
        x = x.permute(0, 3, 1, 2)
        # [B, Fr*C', T, f]
        x = self.L_proj(x)
        # [B, Fr, C', Tr]
        x = x.reshape(x.shape[0], self.groups, self.out_channels, -1)
        x -= torch.mean(x, axis=-1, keepdims=True)
        out = x/(torch.std(x, axis=-1, keepdims=True) + 1e-5)
        # [B, C', C']
        out = torch.einsum('abci,abdi->abcd', out, out)
        # [B, C'*(C'-1)/2]
        out = torch.masked_select(out, self.mask).reshape(batch_size, -1)
        out = out / num_locations
        return out

class AttentivePooling(torch.nn.Module):
    """
    Mean and Standard deviation attentive pooling
    """
    def __init__(self, num_channels, num_freqs=10, attention_channels=128, global_context=False):
        """

        """
        # TODO Make global_context configurable (True/False)
        # TODO Make convolution parameters configurable
        super(AttentivePooling, self).__init__()
        in_factor = 3 if global_context else 1
        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(num_channels * num_freqs * in_factor, attention_channels, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(attention_channels),
            torch.nn.Tanh(),
            torch.nn.Conv1d(attention_channels, num_channels * num_freqs, kernel_size=1),
            torch.nn.Softmax(dim=2),
        )
        self.global_context = global_context
        self.gc = MeanStdPooling()

    def new_parameter(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        """

        :param x: [B, C*F, T]
        :return:
        """
        if len(x.shape) == 4:
            # [B, C, F, T]
            x = x.permute(0, 1, 3, 2)
            # [B, C*F, T]
            x = x.flatten(start_dim=1, end_dim=2)
        if self.global_context:
            w = self.attention(torch.cat([x, self.gc(x).unsqueeze(2).repeat(1, 1, x.shape[-1])], dim=1))
        else:
            w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        rh = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-9) )
        x = torch.cat((mu, rh),1)
        x = x.view(x.size()[0], -1)
        return x


class AttentiveStatsPool(torch.nn.Module):
    """
    Attentive weighted mean and standard deviation pooling.
    """
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """

        :param x:
        :return:
        """
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class GruPooling(torch.nn.Module):
    """
    Pooling done by using a recurrent network
    """
    def __init__(self, input_size, gru_node, nb_gru_layer):
        """

        :param input_size:
        :param gru_node:
        :param nb_gru_layer:
        """
        super(GruPooling, self).__init__()
        self.lrelu_keras = torch.nn.LeakyReLU(negative_slope = 0.3)
        self.bn_before_gru = torch.nn.BatchNorm1d(num_features = input_size)
        self.gru = torch.nn.GRU(input_size = input_size,
                                hidden_size = gru_node,
                                num_layers = nb_gru_layer,
                                batch_first = True)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        return x


def new_parameter(*size):
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


def innerKeyValueAttention(k_proj, query, key, value):
    d_k = k_proj.size(-1)
    proj_key = torch.einsum('ijkl,klm->ijkm', key, k_proj) / math.sqrt(d_k)
    scores = torch.einsum('ijkl,klm->ijkm', proj_key, query) / math.sqrt(d_k)
    scores = scores.reshape(key.shape[0], key.shape[1], -1)
    p_attn = torch.nn.functional.softmax(scores, dim = -2)
    weighted_vector = value * p_attn
    mu = torch.sum(weighted_vector, dim=1)
    rh = torch.sqrt( ( torch.sum((value**2) * p_attn, dim=1) - mu**2 ).clamp(min=1e-9) )
    x = torch.cat((mu, rh),1)
    x = x.view(x.size()[0], -1)
    return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 encoder_size,
                 heads_number,
                 query_size,
                 channel_attention,
                 non_linear,
                 contextual_key,
                 contextual_query,
                 estimate_lambda):
        super(MultiHeadAttention, self).__init__()
        self.channel_attention = channel_attention
        self.non_linear = non_linear
        self.contextual_key = contextual_key
        self.contextual_query = contextual_query
        self.estimate_lambda = estimate_lambda
        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number
        if self.channel_attention:
            self.attention_depth = self.head_size
        else:
            self.attention_depth = 1
        self.heads_number = heads_number
        self.query_size = query_size
        self.k_proj_W = new_parameter(self.heads_number, self.head_size, self.query_size)
        self.query = new_parameter(self.heads_number, self.query_size, self.attention_depth)
        
        if self.contextual_key:
            self.uk_W = new_parameter(self.heads_number, self.head_size*2, self.query_size)

        if self.contextual_query:
            self.u_q1W = new_parameter(self.heads_number, self.head_size*2, self.query_size)
            self.u_q2 = new_parameter(self.heads_number, self.query_size, self.attention_depth)
 
        if self.estimate_lambda:
            self.vhq = new_parameter(self.heads_number, self.attention_depth, self.query_size, 1)
            self.vhk = new_parameter(self.heads_number, self.query_size, 1)
            self.vcq = new_parameter(self.heads_number, self.attention_depth, self.query_size, 1)
            self.vck = new_parameter(self.heads_number, self.query_size, 1)

        if self.non_linear:
            self.k_proj_b = new_parameter(self.heads_number, self.query_size)
            #self.uk_b = new_parameter(self.heads_number, self.query_size)
            self.q_b = new_parameter(self.heads_number, self.attention_depth)
            if self.contextual_query:
                self.u_q1b = new_parameter(self.heads_number, self.query_size)
                self.bnc = torch.nn.BatchNorm2d(self.query_size)
            self.relu = torch.nn.ReLU()
            self.bnk = torch.nn.BatchNorm2d(self.query_size)
            self.tanh = torch.nn.Tanh()

    def getHeadsContextVectors(self,ht):
        B, T, C = ht.shape
        N = self.heads_number
        D = self.head_size
        if self.contextual_key:
            c = torch.cat([torch.mean(ht, dim=1).unsqueeze(-1), torch.std(ht, dim=1).unsqueeze(-1)], dim=2)
            c = c.reshape(B, -1)
            head_c = c.reshape(B, 1, self.heads_number, self.head_size*2)
            head_ck = torch.einsum('ijkl,klm->ijkm', head_c, self.uk_W)
        head_key = ht.reshape(B,-1, self.heads_number, self.head_size)
        value = ht.reshape(B,-1, N, D)
        head_c = c.reshape(B, 1, self.heads_number, self.head_size*2)
        head_key = torch.einsum('ijkl,klm->ijkm', head_key, self.k_proj_W)
        #if self.non_linear:
            #head_key += self.k_proj_b
            #head_ck += self.uk_b
        lambda_k = 0.5
        lambda_q = 0.5
        head_cq = self.query
        if self.contextual_query:
            head_cq = torch.einsum('ijkl,klm->ijkm', head_c, self.u_q1W)
            if self.non_linear:
                head_cq += self.u_q1b
                head_cq = self.tanh(self.bnc(self.relu(head_cq.permute(0, 3, 1, 2)))).permute(0, 2, 3, 1)
            head_cq = torch.einsum('ijkl,klm->ijkm', head_cq, self.u_q2).unsqueeze(-2)
            if self.estimate_lambda:
                lambda_k = torch.sigmoid(torch.einsum('ijkl,klm->ijkm', head_key, self.vhk) \
                    + torch.einsum('ijkl,klm->ijkm', head_ck, self.vck))
                lambda_q = torch.sigmoid(torch.einsum('ijk,ikjl->il', self.query, self.vhq) \
                    + torch.einsum('ijklm,kmln->ijkn', head_cq, self.vcq)).unsqueeze(-1)
            head_cq = (1 - lambda_q) * self.query + lambda_q * head_cq
            head_key = (1 - lambda_k) * head_key + lambda_k * head_ck
        if self.non_linear:
            head_key += self.k_proj_b
            head_key = self.tanh(self.bnk(self.relu(head_key.permute(0, 3, 1, 2)))).permute(0, 2, 3, 1)
        if self.contextual_query:
            scores = torch.einsum('ijkl,ijklm->ijkm', head_key, head_cq) / math.sqrt(self.query_size)
        else:
            scores = torch.einsum('ijkl,klm->ijkm', head_key, head_cq) / math.sqrt(self.query_size)
        if self.non_linear:
            scores += self.q_b
        scores = scores.reshape(scores.shape[0], scores.shape[1], -1, self.attention_depth)
        p_attn = torch.nn.functional.softmax(scores, dim = 1)
        weighted_vector = value * p_attn
        mu = torch.sum(weighted_vector, dim=1)
        rh = torch.sqrt( ( torch.sum((value**2) * p_attn, dim=1) - mu**2 ).clamp(min=1e-9) )
        x = torch.cat((mu, rh),1)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)
        return headsContextVectors.reshape(headsContextVectors.size(0),-1)


class HeadDedicatedQueryAttention(torch.nn.Module):
    def __init__(self,
                 encoder_size,
                 num_heads,
                 query_size,
                 channel_attention,
                 non_linear,
                 contextual_key,
                 contextual_query,
                 estimate_lambda):
        super(HeadDedicatedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.utteranceAttention = MultiHeadAttention(encoder_size,
                                                     num_heads,
                                                     query_size,
                                                     channel_attention,
                                                     non_linear,
                                                     contextual_key,
                                                     contextual_query,
                                                     estimate_lambda)

    def forward(self, x):
        if len(x.shape) == 4:
            # [B, C, F, T]
            x = x.permute(0, 1, 3, 2)
            # [B, C*F, T]
            x = x.flatten(start_dim=1, end_dim=2)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], 3, -1)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        utteranceRepresentation = self.utteranceAttention(x)
        return utteranceRepresentation

