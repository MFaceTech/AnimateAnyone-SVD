import os, io, csv, math, random, cv2, sys
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from decord import VideoReader

sys.path.append("..")
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def make_train_dataset(args):
    if args.dataset_type == "TikTok":
        dataset = TikTok(
            args.csv_path,
            args.video_folder,
            sample_size=[args.height, args.width],
            sample_n_frames=args.num_frames, )
    elif args.dataset_type == "UBC_Fashion":
        dataset = UBC_Fashion(
            args.csv_path,
            args.video_folder,
            sample_size=[args.height, args.width],
            sample_n_frames=args.num_frames, )
    else:
        raise AssertionError("The dataset_type is not supported.")
    return dataset

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255

class TikTok(Dataset):
    def __init__(self,
                 csv_path, 
                 video_folder,
                 sample_size=[512, 512], 
                 sample_stride=4, 
                 sample_n_frames=24,
                 is_image=False
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"video nums: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[0],sample_size[1]]),
            # transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return self.length
    
    def get_batch(self,idx):
        video_dict = self.dataset[idx]
        folder_id, folder_name = video_dict['folder_id'], video_dict['folder_name']
        
        video_dir    =    os.path.join(self.video_folder, folder_name, f"{folder_name}.mp4")
        video_pose_dir =  os.path.join(self.video_folder, folder_name, f"{folder_name}_dwpose.mp4")
        
        video_reader = VideoReader(video_dir)
        video_reader_pose = VideoReader(video_pose_dir)
        
        assert len(video_reader) == len(video_reader_pose), f"len(video_reader) != len(video_reader_pose) in video {idx}"
        
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
            
            
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        pixel_values_pose = torch.from_numpy(video_reader_pose.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        del video_reader_pose
        
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]

        motion_values = 127
        
        return pixel_values, pixel_values_pose, motion_values
    
    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pose_pixel_values, motion_values = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        pose_pixel_values = self.pixel_transforms(pose_pixel_values)
        sample = dict(
            pixel_values=pixel_values, 
            pose_pixel_values=pose_pixel_values,
            motion_values=motion_values
        )
        return sample

    
class UBC_Fashion(Dataset):
    def __init__(self,
                 csv_path, 
                 video_folder,
                 sample_size=[512, 512], 
                 sample_stride=4, 
                 sample_n_frames=14,
                 is_image=False, 
                 is_train=True,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"video nums: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        self.is_train = is_train
        self.spilt = 'train' if self.is_train else 'test'

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([sample_size[0],sample_size[1]]),
            # transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return self.length
    
    def get_batch(self,idx):
        video_dict = self.dataset[idx]
        folder_id, folder_name = video_dict['folder_id'], video_dict['folder_name']
        
        video_dir    =    os.path.join(self.video_folder, self.spilt, f"{folder_name}.mp4")
        video_pose_dir =  os.path.join(self.video_folder, self.spilt+"_dwpose", f"{folder_name}.mp4")
        
        video_reader = VideoReader(video_dir)
        video_reader_pose = VideoReader(video_pose_dir)
        
        assert len(video_reader) == len(video_reader_pose), f"len(video_reader) != len(video_reader_pose) in video {idx}"
        
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
            
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        pixel_values_pose = torch.from_numpy(video_reader_pose.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        del video_reader_pose
        
        motion_values = 127
        
        return pixel_values, pixel_values_pose, motion_values
    
    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pose_pixel_values, motion_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length-1)
        
        pixel_values = self.pixel_transforms(pixel_values)
        pose_pixel_values = self.pixel_transforms(pose_pixel_values)
       
        sample = dict(
            pixel_values=pixel_values, 
            pose_pixel_values=pose_pixel_values,
            motion_values=motion_values
            )
        
        return sample


if __name__ == "__main__":
    dataset = UBC_Fashion(
        csv_path="UBC_train_info.csv",
        video_folder="ubc-fashion",
        sample_size=[512, 512], 
        sample_stride=4, 
        sample_n_frames=16,
        is_image=False
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8,)
    for idx, batch in enumerate(dataloader):
        print("motion_values: ", idx, batch["motion_values"].shape)
        # print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)