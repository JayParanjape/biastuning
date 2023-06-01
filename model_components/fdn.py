import torch.nn as nn


class FDN_Conv_Block(nn.Module):
    def __init__(self, in_channels, norm_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, norm_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(norm_channels),
            nn.Conv2d(norm_channels, norm_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(norm_channels),
            nn.Conv2d(norm_channels, norm_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class FDN(nn.Module):
    def __init__(self, norm_nc, input_nc, reduction_factor=4):
        super().__init__()
        ks = 3
        self.norm = nn.BatchNorm2d(norm_nc)
        self.conv_gamma = nn.ModuleList()
        self.conv_beta = nn.ModuleList()
        
        for i in range(reduction_factor):
            if i==0:
                self.conv_gamma.append(FDN_Conv_Block(input_nc, norm_nc))
            else:
                self.conv_gamma.append(FDN_Conv_Block(norm_nc, norm_nc))

        for i in range(reduction_factor):
            if i==0:
                self.conv_beta.append(FDN_Conv_Block(input_nc, norm_nc))
            else:
                self.conv_beta.append(FDN_Conv_Block(norm_nc, norm_nc))
                
    def forward(self, x, original_image):
        normalized = self.norm(x)
        for i in range(len(self.conv_gamma)):
            if i==0:
                gamma = self.conv_gamma[0](original_image)
            else:
                gamma = self.conv_gamma[i](gamma) + gamma
            gamma = nn.functional.max_pool2d(gamma, 2)
                
        for i in range(len(self.conv_beta)):
            if i==0:
                beta = self.conv_beta[0](original_image)
            else:
                beta = self.conv_beta[i](beta) + beta
            beta = nn.functional.max_pool2d(beta, 2)

        gamma = gamma.view(normalized.shape)
        beta = beta.view(normalized.shape)
        out = normalized * (1 + gamma) + beta
        return out

