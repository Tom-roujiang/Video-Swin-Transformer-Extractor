import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
import numpy as np
import os 
import math
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
import argparse
import h5py
import cv2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 


'''
initialize a SwinTransformer3D model
'''
# model = SwinTransformer3D().to('cuda')
# print(model)

# dummy_x = torch.rand(1, 3, 32, 224, 224).to('cuda')
# logits = model(dummy_x)
# print(logits.shape)

'''
load the pretrained weight

1. git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git
2. move all files into ./Video-Swin-Transformer

'''
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')

'''
use the pretrained SwinTransformer3D as feature extractor
'''


def load_frame(hdf5_file, frame_file):

    data = hdf5_file[frame_file]
    data = np.array(data)
    data = Image.fromarray(data)
    # data = Image.open(data) 
    data = data.resize((224, 224), Image.LANCZOS)

    # image = data.astype(np.uint8)
    # resized_image = cv2.resize(image, (224, 224))
    # data = data.resize((224, 224), Image.LANCZOS)
    # data = resized_image.astype(np.float32)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data


def load_rgb_batch(frames_dir, rgb_files, 
                   frame_indices, hdf5_file):  
    # 表示一个批次中的所有视频片段及其每一帧的 RGB 图像。
    batch_data = np.zeros(frame_indices.shape + (224,224,3)) # b x clip length x 224 x 224 x 3  
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            frame_path = '{}/{}'.format(frames_dir, rgb_files[frame_indices[i][j]])
            batch_data[i,j,:,:,:] = load_frame(hdf5_file, frame_path)

            # batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, 
            #     rgb_files[frame_indices[i][j]]))
    return batch_data


def forward_batch(b_data, model):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)     # b x 3 x Temporial(length of a clip) x 224 x 224
    with torch.no_grad():
        model.eval()
        b_data = Variable(b_data.cpu()).float()
        b_features = model(b_data)
    b_features = b_features.data.cpu().numpy()[:,:,0,0,0]
    return b_features


def extract_feature(args_item):
    video_dir, output_dir, batch_size, task_id = args_item
    mode='rgb'
    chunk_size = 16          # 每个clip包含帧的数量 
    frequency=16             # frequency表示帧采样的频率
    sample_mode='oversample'
    video_name=video_dir.split("/")[-1]
    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    save_file = '{}_{}.npy'.format(video_name, "swin")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        pass

    else:  

        # setup the model  
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        # model.eval()
        model = model.backbone.to('cpu')
        
        f = h5py.File("/Users/liyun/Desktop/h5py-test/mytestfile1.hdf5", 'r')
        rgb_files = [i for i in f[video_dir] if i.endswith('jpg')]
        # rgb_files = [i for i in os.listdir(video_dir) if i.endswith('jpg')]
        rgb_files.sort(key=lambda x:int(x.split("_")[1].split(".")[0]))
        frame_cnt = len(rgb_files)

        assert(frame_cnt > chunk_size)
    
        

        # 处理最后一段不完整的情况，如果最后一段不满，则复制最后一帧直到填满分段
        # 第二种策略：直接丢弃不满的最后一段
        
        # 策略1： copy ----------------------------------------------------------
        # clipped_length = math.ceil(frame_cnt / chunk_size)  # 向上取整, 表示一共有多少clip
        # copy_length = (clipped_length * frequency) - frame_cnt  # The start of last chunk
        # if copy_length!=0:
        #     copy_img=[rgb_files[frame_cnt-1]]*copy_length
        #     rgb_files=rgb_files+copy_img

        # 策略2： discard ----------------------------------------------------------
        clipped_length = math.floor(frame_cnt / chunk_size)
        discard_length = frame_cnt - (clipped_length * chunk_size)
        if discard_length != 0:
            rgb_files = rgb_files[:-discard_length]
            # clipped_length -= 1

        frame_indices = [] # Frames to chunks
        for i in range(clipped_length):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])

        frame_indices = np.array(frame_indices)   # size = (clips, frame_per_clip)
        chunk_num = frame_indices.shape[0]        

        batch_num = int(np.ceil(chunk_num / batch_size))   # number of batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)   # split into batches by the first dim

        # full_features = [[] for i in range(10)]
        # # 
        for batch_id in tqdm(range(batch_num)):       
            batch_data = load_rgb_batch(video_dir, rgb_files, frame_indices[batch_id], f)
            full_features = forward_batch(batch_data, model)

        np.save(os.path.join(output_dir,save_file), full_features)

        print('{} done: {} / {}, {}'.format(
            video_name, frame_cnt, clipped_length, full_features.shape))   # full_features.shape = (10, b, 1024)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rgb",type=str)
    # parser.add_argument('--load_model',default="feature_extract/model_rgb.pth", type=str)
    # parser.add_argument('--input_dir', default="UCF_Crime_Frames",type=str)
    parser.add_argument('--input_hdf5', default="/Users/liyun/Desktop/h5py-test/mytestfile1.hdf5",type=str)
    parser.add_argument('--output_dir',default="UCF-Swin", type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    # parser.add_argument('--sample_mode', default="oversample",type=str)
    parser.add_argument('--frequency', type=int, default=16)
    args = parser.parse_args()

    vid_list=[]

    file_path = args.input_hdf5
    f = h5py.File(file_path, 'r')



    
    for key in f.keys():
        type_path = f"/{key}"   # save the path of the type
        for video_name in f[type_path]:
            save_file = '{}_{}.npy'.format(video_name, "swin")
            if save_file in os.listdir(os.path.join(args.output_dir)):
                print("{} has been extracted".format(save_file))
            else:
                video_path = f"{type_path}/{video_name}"
                vid_list.append(video_path)    

    # for videos in os.listdir(args.input_dir):
    #     for video in os.listdir(os.path.join(args.input_dir,videos)):
    #         save_file = '{}_{}.npy'.format(video, "swintrans")
    #         if save_file in os.listdir(os.path.join(args.output_dir)):
    #             print("{} has been extracted".format(save_file))
    #         else:
    #             vid_list.append(os.path.join(args.input_dir,videos,video))
    
    nums=len(vid_list)
    print("leave {} videos".format(nums))
    # pool = Pool(4)
    # pool.map(run, zip([args.load_model]*nums, vid_list, [args.output_dir]*nums,[args.batch_size]*nums,range(nums)))
    for item in zip(vid_list, [args.output_dir]*nums,[args.batch_size]*nums,range(nums)):
        print(item)
        extract_feature(item)






