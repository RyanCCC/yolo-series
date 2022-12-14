import numpy as np
from PIL import Image
from pathlib import Path


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

def check_suffix(file='yolov5s.pth', suffix = ('.pth'), msg = '' ):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            assert Path(f).suffix.lower() in suffix, f"{msg}{f} acceptable suffix is {suffix}"

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
        
def download_weights(backbone, phi, model_dir="./checkpoints"):
    import os
    from torch.hub import load_state_dict_from_url
    if backbone == "cspdarknet":
        backbone = backbone + "_" + phi
    
    download_urls = {
        "convnext_tiny"         : "https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/convnext_tiny_1k_224_ema_no_jit.pth",
        "convnext_small"        : "https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/convnext_small_1k_224_ema_no_jit.pth",
        "cspdarknet_s"          : 'https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/cspdarknet_s_backbone.pth',
        'cspdarknet_m'          : 'https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/cspdarknet_m_backbone.pth',
        'cspdarknet_l'          : 'https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/cspdarknet_l_backbone.pth',
        'cspdarknet_x'          : 'https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/cspdarknet_x_backbone.pth',
        'swin_transfomer_tiny'  : "https://github.com/RyanCCC/YOLOSeries/releases/download/v1.0.0/swin_tiny_patch4_window7.pth",
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)