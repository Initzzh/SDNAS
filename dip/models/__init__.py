from .net_indi import Unet as unet_indi
from.S2Snet import S2Snet
from .net_indi import s2s_test
from .net_indi import s2s_test_MIMO
from .net_indi import s2s_test_first
from .net_indi import UnetScale as unet_indi_scale
from .net_indi import UnetConnect as unet_indi_connect



def get_net_indi(args, indi):

    NET_TYPE = args.net_type
    input_depth = 1 if args.gray else 3
    n_channels = 1 if args.gray else 3

    act_fun = 'soft'
    if NET_TYPE == 'unet_indi':
        print("[*] Net_type : unet_indi")
        net = unet_indi(indi, input_depth)
        print("net",net)
    elif NET_TYPE == 'unet_indi_scale':
        print("[*] Net_type : unet_indi_scale")
        net = unet_indi_scale(indi, input_depth)
        # print("net",net)
    elif NET_TYPE == 'unet_indi_connect':
        print("[*] Net_type : unet_indi_connect")
        net = unet_indi_connect(indi, input_depth)
    elif NET_TYPE == 's2s':
        print("[*] Net_type : s2s")
        net = S2Snet(input_depth, n_channels, act_type=act_fun)

    elif NET_TYPE == 's2s_test':
        print("[*] Net_type : s2s_test")
        net = s2s_test(None)
        print(net)
    elif NET_TYPE == 's2s_test_first':
        print("[*] Net_type : s2s_test_frist")
        net = s2s_test_first(None)
        print(net)

    elif NET_TYPE == 's2s_test_MIMO':
        print("[*] Net_type : s2s_test_MIMO")
        net = s2s_test_MIMO(None)
    else:
        assert False
    
    return net




    

    

    