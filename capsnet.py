import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    """
    PrimaryCaps 层不涉及到动态路由算法。在 PrimaryCaps 层中，每个胶囊的输出向量都是通过一个独立的卷积层生成的，与其他胶囊无关，因此不需要进行动态路由操作。
    """
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        """

        :param x: [B, in_channels, input_height, input_width]
        :type x:
        :return: [B, num_routes, 8]
        :rtype:
        """
        # out_channels=32
        # list of tensor. tensor shape [B, out_channels, output_height, output_width]. num_capsules个元素。
        u = [capsule(x) for capsule in self.capsules]

        # [B, num_capsules, out_channels, output_height, output_width]
        u = torch.stack(u, dim=1)

        # [B, num_routes, 8]
        # todo num_routes= num_capsules * out_channels* output_height???
        # 将多个cnn的的结果concat到一起，每个cnn的结果又由多个channel结果组成
        u = u.view(x.size(0), self.num_routes, -1)

        # 虽然 PrimaryCaps 层不涉及到动态路由算法，但是在实现中仍然使用了 squash 函数
        # [B, num_routes, 8]
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        """

        :param x: [B, self.num_routes, in_channels]. 2048个vector，每个vector是8dimension。
        :type x:
        :return:
        :rtype:
        """
        batch_size = x.size(0)
        # [B, self.num_routes, self.num_capsules, in_channels, 1]
        # 通过copy，每个capsule的输入是相同的
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        # batch里的每个样本共享self.W。
        W = torch.cat([self.W] * batch_size, dim=0)

        # x从in_channels维变成了u_hat的out_channels维
        # u_hat:[B, self.num_routes, self.num_capsules, out_channels, 1]
        # W:[B, self.num_routes, self.num_capsules, out_channels, in_channels]
        # x:[B, self.num_routes, self.num_capsules, in_channels, 1]
        u_hat = torch.matmul(W, x)

        # [1, self.num_routes, self.num_capsules, 1]
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        # 使用动态路由算法计算输出向量 v_j，并将其返回。
        num_iterations = 3
        for iteration in range(num_iterations):
            # [1, self.num_routes, self.num_capsules, 1]
            c_ij = F.softmax(b_ij, dim=1)

            # list of element [1, self.num_routes, self.num_capsules, 1]. 100个元素。batch里的每个样本共享bij、cij。
            c_ij = [c_ij] * batch_size
            c_ij = torch.cat(c_ij, dim=0)  # [B, self.num_routes, self.num_capsules, 1]
            c_ij = c_ij.unsqueeze(4) # [B, self.num_routes, self.num_capsules, 1, 1]

            # s_j:[B, self.num_routes, self.num_capsules, out_channels, 1]
            # c_ij: [B, self.num_routes, self.num_capsules, 1, 1]
            # u_hat: [B, self.num_routes, self.num_capsules, out_channels, 1]
            s_j = c_ij * u_hat
            # [B, 1, num_capsules, out_channels, 1]
            s_j = s_j.sum(dim=1, keepdim=True)

            # [B, 1, num_capsules, out_channels, 1]
            v_j = self.squash(s_j)

            # 这个地方不加if条件限制也行，不加，就会多算而已
            if iteration < num_iterations - 1:
                # [B, self.num_routes, self.num_capsules, 1, out_channels]
                u_hat_trans = u_hat.transpose(3, 4)
                # [B, num_routes, num_capsules, out_channels, 1]
                v_j_concat = torch.cat([v_j] * self.num_routes, dim=1)

                # a_ij 是 u_hat 和 v_j 之间的点积，表示 v_j 对 u_hat 的影响程度。
                # a_ij：[B, self.num_routes, self.num_capsules, 1, 1]
                # u_hat_trans：[B, self.num_routes, self.num_capsules, 1, out_channels]
                # v_j_concat：[B, num_routes, num_capsules, out_channels, 1]
                a_ij = torch.matmul(u_hat_trans, v_j_concat)

                # [1, self.num_routes, self.num_capsules, 1]
                # todo 为什么batch做平均呢？
                # 这样做的目的是将每个样本的 a_ij 值进行平均，得到一个共享的动态路由权重。在反向传播时，这个共享的动态路由权重可以同时对所有样本进行梯度更新，从而提高模型的训练效率。
                a_ij = a_ij.squeeze(4).mean(dim=0, keepdim=True)

                # [1, self.num_routes, self.num_capsules, 1]
                b_ij = b_ij + a_ij

        # [B, num_capsules, out_channels, 1]
        v_j = v_j.squeeze(1)

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        self.config = None
        if config:
            self.config = config
            # pc: primary capsule的缩写；dc：digit capsule的缩写。
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        """

        :param data: [B, cnn_in_channels, input_height, input_width]
        :type data:
        :return:
        :rtype:
        """
        # [B, cnn_out_channels, input_height, input_width]
        conv_out = self.conv_layer(data)

        # [B, pc_num_routes, 8]
        primary_out = self.primary_capsules(conv_out)

        # [B, dc_num_capsules, dc_out_channels, 1]
        output = self.digit_capsules(primary_out)

        # reconstructions:[B, cnn_out_channels, input_height, input_width]
        # masked:[B, out_channels]
        reconstructions, masked = self.decoder(output, data)

        return output, reconstructions, masked

    def loss(self, data, pred, target, reconstructions):
        """

        :param data: 胶囊网络的输入值。[B, cnn_in_channels, input_height, input_width]
        :type data:
        :param pred: 胶囊网络的预估输出值。[B, num_capsules, 16, 1]
        :type pred:
        :param target: target。[B, 10]
        :type target:
        :param reconstructions: decoder重建值。[B, cnn_in_channels, input_height, input_width]
        :type reconstructions:
        :return:
        :rtype:
        """
        return self.margin_loss(pred, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, pred, labels, size_average=True):
        batch_size = pred.size(0)
        pred_norm = (pred ** 2).sum(dim=2, keepdim=True)

        # 每个胶囊的l2 norm
        # [B, num_capsules, 1, 1]
        v_c = torch.sqrt(pred_norm)

        # [B, num_capsules]
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        lambda_value = 0.5
        # [B, num_capsules]
        loss = labels * left + lambda_value * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        # scalar
        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005
