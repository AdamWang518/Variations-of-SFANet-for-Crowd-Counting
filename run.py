import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import torch
from torchvision import transforms

from crowd import Crowd
from models import M_SFANet_UCF_QNRF
from matplotlib import cm as CM

def resize(density_map, image):
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    image= image[0]
    print(density_map.shape)
    result_img = np.zeros((density_map.shape[0]*2, density_map.shape[1]*2))
    for i in range(result_img.shape[0]):
        for j in range(result_img.shape[1]):
            result_img[i][j] = density_map[int(i / 2)][int(j / 2)] / 4
    result_img  = result_img.astype(np.uint8, copy=False)
    return result_img

def vis_densitymap(o, den, cc, img_path):
    fig=plt.figure()
    columns = 2
    rows = 1
    # 确保 o 是一个 NumPy 数组
    if isinstance(o, torch.Tensor):
        o = o.squeeze().numpy()
    else:
        o = np.squeeze(o)
    X = np.transpose(o, (1, 2, 0))
    
    summ = int(np.sum(den))
    
    den = resize(den, o)
    
    for i in range(1, columns*rows +1):
        # image plot
        if i == 1:
            img = X
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.imshow(img)
            
        # Density plot
        if i == 2:
            img = den
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.text(1, 80, 'M-SegNet* Est: '+str(summ)+', Gt:'+str(cc), fontsize=7, weight="bold", color = 'w')
            plt.imshow(img, cmap=CM.jet)
    
    filename = img_path.split('/')[-1]
    filename = filename.replace('.jpg', '_heatpmap.png')
    print('Save at', filename)
    plt.savefig('seg_'+filename, transparent=True, bbox_inches='tight', pad_inches=0.0, dpi=200)


# Simple preprocessing.
trans = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])

# An example image with the label = 1236.
img = Image.open("./images/test.jpeg").convert('RGB')
height, width = img.size[1], img.size[0]
height = round(height / 16) * 16
width = round(width / 16) * 16
img = cv2.resize(np.array(img), (width,height), cv2.INTER_CUBIC)
img = trans(Image.fromarray(img))[None, :]

model = M_SFANet_UCF_QNRF.Model()
# Weights are stored in the Google drive link.
# The model are originally trained on a GPU but, we can also test it on a CPU.
# For ShanghaitechWeights, use torch.load("./ShanghaitechWeights/...")["model"] with M_SFANet.Model() or M_SegNet.Model()
model.load_state_dict(torch.load("best_M-SFANet__UCF_QNRF.pth", 
                                 map_location = torch.device('cpu')))

# Evaluation mode
model.eval()
density_map = model(img)
vis_densitymap(img, density_map.cpu().detach().numpy(), int(torch.sum(density_map).item()), "./images/test.jpeg")

# Est. count = 1168.37 (67.63 deviates from the ground truth)
print(torch.sum(density_map).item())