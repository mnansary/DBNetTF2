# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
from coreLib.dataset import DataSet
from coreLib.utils import create_dir,LOG_INFO
from coreLib.render import createSceneData,backgroundGenerator
from coreLib.format import create_train_data
from tqdm.auto import tqdm
import os
import cv2
import random


def single_data(ds,backgen,dim):
    back,page=createSceneData(ds,backgen)
    back=cv2.resize(back,(dim,dim))
    page=cv2.resize(page,(dim,dim),fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
    gt,mask,thresh_map,thresh_mask=create_train_data(page)
    return back,gt,mask,thresh_map,thresh_mask

def main(args):
    data_dir=args.data_dir
    save_dir=args.save_dir
    n_data=int(args.n_data)
    dim=int(args.dim)
    # save
    save_dir=create_dir(save_dir,"dbsynth")
    img_dir=create_dir(save_dir,"image")
    gt_dir=create_dir(save_dir,"gt")
    mask_dir=create_dir(save_dir,"mask")
    thresh_mask_dir=create_dir(save_dir,"thresh_mask")
    thresh_map_dir=create_dir(save_dir,"thresh_map")
    
    # sources
    ds=DataSet(data_dir)
    backgen=backgroundGenerator(ds)
    LOG_INFO(save_dir)



    for fiden in tqdm(range(n_data)):
        try:
            img,gt,mask,thresh_map,thresh_mask=single_data(ds,backgen,dim)
            # save
            cv2.imwrite(os.path.join(img_dir,f"{fiden}.png"),img)
            cv2.imwrite(os.path.join(gt_dir,f"{fiden}.png"),gt)
            cv2.imwrite(os.path.join(mask_dir,f"{fiden}.png"),mask)
            cv2.imwrite(os.path.join(thresh_map_dir,f"{fiden}.png"),thresh_map)
            cv2.imwrite(os.path.join(thresh_mask_dir,f"{fiden}.png"),thresh_mask)
            fiden+=1
        except Exception as e:
            pass


if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic DBNet Data Creation Script")
    parser.add_argument("data_dir", help="Path to base data under source")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--dim",required=False,default=512,help ="dimension of the image : default=512")
    parser.add_argument("--n_data",required=False,default=5000,help ="number of data to create : default=5000")
    
    args = parser.parse_args()
    main(args)
