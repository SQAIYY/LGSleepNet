import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import PatchEmbed, Mlp, DropPath,GluMlp
import torchsummary

class SoftLayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SoftLayer, self).__init__()

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.soft(x)
        return x * y
class AttenBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes,
                 reduction=1):
        super(AttenBasicBlock, self).__init__()
        self.soft = SoftLayer(planes, reduction)
    def forward(self, x):
        out = self.soft(x)
        return out
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p),  x.size(-1)).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, local_feat, global_feat):
        if len(global_feat.shape) == 1:  # 方式出现训练最后一个step时，出现v是一维的情况
            global_feat = torch.unsqueeze(global_feat, 0)
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        # (f_l * f_g * f_g) / (f_g * f_g)
        projection = projection / \
            (global_feat_norm * global_feat_norm).view(-1, 1, 1)
        # 正交分量
        print("projection",projection.shape)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-2)

        return global_feat, orthogonal_comp
class Attention(nn.Module):
    def __init__(self, dim, num_heads=5, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
from einops.layers.torch import Rearrange

class Block(nn.Module):

    def __init__(self, dim, num_heads=5, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GluMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        #print("11111",self.norm2(x).shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    #result = torch.tensor(result)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return result.to(device)
    #return result


class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 100
        self.decoder_depth = 2
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.dim,
                drop=0.2,
                attn_drop=0.2,
            )
            for i in range(self.decoder_depth)])

        self.to_patch_embedding = nn.Sequential(
            #Rearrange('b c (n p) -> b n (p c)', p=100),
            nn.Linear(100, 100),
        )
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))  # 1, 1, 8000
        self.fc2 = nn.Linear(3100, 5)
        self.conv =  nn.Conv1d(1, 256, kernel_size=1, stride=1, bias=False)
        self.GELU1 = nn.GELU()

        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x):
        # x -> bz x 59 x 8000
        # print(x.shape)
        x = x.reshape(-1,30,100)
        x = self.to_patch_embedding(x)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # bz x 1 x 8000
        x = torch.cat((x, cls_token), 1)  # bz x 60 x 8000
        x = x + get_positional_embeddings(31,100)
        #print(x.shape)
        x = self.blocks(x)
        cls_token = x[:, -1:, :]
        cls_token = self.conv(cls_token)
        cls_token = self.GELU1(cls_token)
        return cls_token
class EEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=3, bias=False, padding=12),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.Conv1d(64, 64, kernel_size=16, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            #nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            #nn.BatchNorm1d(128),
            #self.GELU,

            nn.Conv1d(128, 256, kernel_size=4, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            #nn.Dropout(drate)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class EEGCNNVIT(nn.Module):
    def __init__(self):
        super().__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.RELU = nn.ReLU()
        self.features1 = EEGTransformer()
        self.features2 = EEGCNN()
        self.dropout = nn.Dropout(drate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 5)
        self.tanh = nn.Tanh()
        self.gem_pool = GeM()
        self.orthogonal_fusion = OrthogonalFusion()
        self.softmax = nn.Softmax(dim=1)
        self.RELU1 = nn.ReLU()
        afr_reduced_cnn_size = 2
        self.inplanes = 2
        self.AFR = self._make_layer(AttenBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block blocks=1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 128, 30
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(planes, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.features1(x) #全局
        x2 = self.features2(x)#局部 [2, 128, 64]
        global_feat = self.fc_1(self.gem_pool(x1).squeeze()) #[2, 128]
        global_feat, orthogonal_comp = self.orthogonal_fusion(x2, global_feat)  # [2, 256, 64]
        orthogonal_comp = self.fc_2(self.gem_pool(orthogonal_comp).squeeze())  # [2, 128]
        orthogonal_comp = orthogonal_comp.unsqueeze(-2)

        if len(orthogonal_comp.shape) == 2:  # 方式出现训练最后一个step时，出现v是一维的情况
            orthogonal_comp = orthogonal_comp.unsqueeze(-2) #torch.unsqueeze(orthogonal_comp, 0)
        feat_cat = torch.cat([global_feat, orthogonal_comp], dim=1)
        #print("feat",feat_cat.shape)
        feat_cat = self.AFR(feat_cat)
        #feat_cat = self.softmax(feat_cat)
        feat_cat_sum = torch.sum(feat_cat, dim=1)
        x = self.fc(feat_cat_sum)
        x = self.RELU1(x)
        x = self.fc1(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGCNNVIT().to(device)
    x = torch.randn((3, 1, 3000)).to(device)
    out = model(x)
    print(out.shape)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
    #device = torch.device('cpu')
    #model.to(device)
    torchsummary.summary(model.cuda(), (1, 3000))