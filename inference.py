import torch
import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

from data.compose import Compose
from data.config import cfg, process_funcs_dict
from modules.solov2 import SOLOV2

import os
from glob import glob
import warnings 
import argparse
warnings.filterwarnings("ignore")

def build_process_pipeline(pipeline_confg):
    assert isinstance(pipeline_confg, list)
    process_pipelines = []
    for pipconfig in pipeline_confg:
        assert isinstance(pipconfig, dict) and 'type' in pipconfig
        args = pipconfig.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            process_pipelines.append(process_funcs_dict[obj_type](**args))
    return process_pipelines

def eval(valmodel_weight, data_path):
    test_pipeline = []
    
    # Using test_pipeline from config instead of test_transforms
    transforms_piplines = build_process_pipeline(cfg.test_config.test_pipeline)
    Multest = process_funcs_dict['Format'](transforms=transforms_piplines)

    test_pipeline.append(process_funcs_dict['LoadImageFromFile']())
    test_pipeline.append(Multest)
    test_pipeline = Compose(test_pipeline)

    model = SOLOV2(cfg, pretrained=valmodel_weight, mode='test')
    model = model.cuda()

    # Process images from data_path
    test_imgpath = data_path + '/*'
    images = glob(test_imgpath)

    for k, imgpath in enumerate(images):
        # Process image through pipeline
        data = test_pipeline(dict(img=imgpath))
        imgs = data['img']
        img = imgs[0].cuda().unsqueeze(0)
        img_info = data['img_metas']
        
        # Run model inference
        with torch.no_grad():
            model_output = model.forward(img=[img], img_meta=[img_info], return_loss=False)
        
        # Enhanced error handling
        try:
            # Get results
            raw_img = cv.cvtColor(cv.imread(imgpath), cv.COLOR_BGR2RGB)
            seg_img = model_output[0][0][0].cpu().numpy()
            classes = [cfg.label[i.item()] for i in model_output[0][1][:3]]
            scores = [round(i.item(), 2) for i in model_output[0][2][:3]]
            
            # Visualize and save results
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))
            ax[0].imshow(raw_img)
            ax[0].set_title('Original Image')
            ax[1].imshow(seg_img)
            ax[1].set_title(f'Seg Result | Class: {classes} | Score: {scores}')
            plt.tight_layout()
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Save output
            out_filepath = "results/" + os.path.basename(imgpath)
            plt.savefig(out_filepath)
            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {imgpath}: {e}. Moving to next image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SOLOv2 Inference')
    parser.add_argument('--epoch', type=int, default=10, help='Model epoch to use for inference')
    parser.add_argument('--data_path', type=str, default="/home/localssk23/SOLOv2_2D/datasets/dummy_2d/test_imgs", 
                        help='Path to test images')
    args = parser.parse_args()
    
    eval(valmodel_weight=f'weights/solov2_resnet18_epoch_{args.epoch}.pth', 
         data_path=args.data_path)

# example usage:
# python inference.py --epoch 10 --data_path /home/localssk23/SOLOv2/datasets/dummy_2d/test_imgs