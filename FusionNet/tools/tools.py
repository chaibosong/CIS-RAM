from nets.hnet import *

hnet = HNet(2)

# pretrained_dict_hnet = torch.load('../weights/model_hnet.pth')
pretrained_hnet_path = "../weights/model_hnet.pth"
hnet.load_state_dict(torch.load(pretrained_hnet_path))