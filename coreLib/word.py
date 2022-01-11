# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 


from tqdm import tqdm
from glob import glob


from .config import config
from .utils import *
tqdm.pandas()



#--------------------
# word functions 
#--------------------
def createHandwritenWords(df,
                         comps):
    '''
        creates handwriten word image
        args:
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components 
        returns:
            img     :       marked word image
            iden    :       the final identifier
    '''
    comps=[str(comp) for comp in comps]
    # select a height
    height=config.comp_dim
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]
    # construct labels
    imgs=[]
    for comp in comps:
        c_df=df.loc[df.label==comp]
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=c_df.iloc[idx,2] 
        # read image
        img=cv2.imread(img_path,0)
        # resize
        h,w=img.shape 
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # mark image
        img=255-img
        imgs.append(img)
        
    img=np.concatenate(imgs,axis=1)
    return img

def createPrintedWords(comps,
                       fonts):
    '''
        creates printed word image
        args:
            comps   :       the list of components
            fonts   :       available font paths 
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
    '''
    
    comps=[str(comp) for comp in comps]
    # select a font size
    font_size=config.comp_dim
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    for idx,comp in enumerate(comps):
        if idx < len(comps)-1 and comps[idx+1] in mods:
            comps[idx]+=comps[idx+1]
            comps[idx+1]=None 
            
    comps=[comp for comp in comps if comp is not None]
    # font path
    font_path=random.choice(fonts)
    font=PIL.ImageFont.truetype(font_path, size=font_size)
    image = PIL.Image.new(mode='L', size=font.getsize("".join(comps)))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text="".join(comps), fill=1, font=font)    
    # word
    img=np.array(image)
    img=stripPads(img,0)
    
    h,w=img.shape 
    width= int(font_size* w/h) 
    
    img=cv2.resize(img,(width,font_size),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img


#-----------------------------------
# wrapper
#----------------------------------
def create_word(iden,
                source_type,
                data_type,
                comp_type,
                ds,
                use_dict=True):
    '''
        creates a marked word image
        args:
            iden                    :       identifier marking value starting
            source_type             :       bangla/english 
            data_type               :       handwritten/printed                  
            comp_type               :       grapheme/number/mixed
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
    '''
        
    
    # set resources
    if source_type=="bangla":
        dict_df  =ds.bangla.dictionary 
        
        g_df     =ds.bangla.graphemes.df 
        
        n_df     =ds.bangla.numbers.df 
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path]
    elif source_type=="english":
        dict_df  =ds.english.dictionary 
        
        g_df     =ds.english.graphemes.df 
        
        n_df     =ds.english.numbers.df 
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]

    # component selection 
    if comp_type=="grapheme":
        # dictionary
        if use_dict:
            # select index from the dict
            idx=random.randint(0,len(dict_df)-1)
            comps=dict_df.iloc[idx,1]
        else:
            # construct random word with grapheme
            comps=[]
            len_word=random.randint(config.min_word_len,config.max_word_len)
            for _ in range(len_word):
                idx=random.randint(0,len(g_df)-1)
                comps.append(g_df.iloc[idx,1])
        df=g_df
    elif comp_type=="number":
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(n_df)-1)
            comps.append(n_df.iloc[idx,1])
        df=n_df
    
    else:
        sdf         =   ds.common.symbols.df
        df=pd.concat([g_df,n_df,sdf],ignore_index=True)
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(df)-1)
            comps.append(df.iloc[idx,1])

    
    # process data
    if data_type=="handwritten":
        img=createHandwritenWords(df=df,comps=comps)
    else:
        img=createPrintedWords(comps=comps,fonts=fonts)
    # warp
    if random_exec(weights=[0.3,0.7]):
        img=warp_data(img)
    # rotate/curve
    if random_exec(weights=[0.5,0.5]):
        if random_exec(weights=[0.5,0.5]):
            angle=random.randint(-90,90)
            img=rotate_image(img,angle)
        else:
            img=curve_data(img)
    
    img[img>0]=iden
    iden+=1
    img=np.squeeze(img)
    # add space
    h,_=img.shape
    w=random.randint(config.word_min_space,config.word_max_space)
    img=np.concatenate([img,np.zeros((h,w))],axis=1)           
            
    return img,iden


    
        
    