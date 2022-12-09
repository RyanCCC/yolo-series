import numpy as np
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
        
def download_weights(phi, model_dir):
    import os
    from torch.hub import load_state_dict_from_url
    
    backbone = "cspdarknet_" + phi
    download_urls = {
        "cspdarknet_n" : 'https://github.com/ryanccc/YOLOSeries/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
        "cspdarknet_s" : 'https://github.com/ryanccc/YOLOSeries/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
        'cspdarknet_m' : 'https://github.com/ryanccc/YOLOSeries/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
        'cspdarknet_l' : 'https://github.com/ryanccc/YOLOSeries/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
        'cspdarknet_x' : 'https://github.com/ryanccc/YOLOSeries/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)