#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary


from ptflops import get_model_complexity_info
from wavemamba_arch import UNet
# from models import Generator_prune as Generator

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = UNet(in_chn=3, wf=32, n_l_blocks=[1, 2, 4], n_h_blocks=[1, 1, 2], ffn_scale=2).to(device)

    macs, params = get_model_complexity_info(m, (3, 256, 256), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#


