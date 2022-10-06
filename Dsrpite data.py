

from utils.datasets import *
import logging
import numpy as np
import pickle as pkl

RES_DIR = "results"
exp_dir = os.path.join(RES_DIR, "CVAE")
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
stream = logging.StreamHandler()
stream.setLevel("INFO")
stream.setFormatter(formatter)
logger.addHandler(stream)
logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))


dataset = DSprites(logger=logger)


img = []
label = []
for i in range(len(dataset)):
    if i % 10000 == 0:
        print(i)
        
    img.append(dataset[i][0])
    # Properties: 'color', 'shape','scale','orientation','posX', 'posY', 
    # Hand-crafted properties: 'distance^2 from origin', 'scaleX = posX*scale', 'scaleY=posY*scale'
    label_tmp = dataset[i][1]
    label_tmp = np.append(label_tmp, [label_tmp[4]**2 + label_tmp[5]**2, label_tmp[4] * label_tmp[2], label_tmp[5] * label_tmp[2]])
    label.append(label_tmp)

with open('DSprites_feature.pkl', 'wb') as f:
    pkl.dump(label, f)
    
with open('DSprotes_img.pkl', 'wb') as f:
    pkl.dump(img, f)
