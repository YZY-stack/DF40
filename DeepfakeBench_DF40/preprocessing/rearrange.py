# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""
After running this code, it will generates a json file looks like the below structure for re-arrange data.

{
    "FaceForensics++": {
        "Deepfakes": {
            "video1": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video1/frame1.png",
                    "/path/to/frames/video1/frame2.png",
                    ...
                ]
            },
            "video2": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video2/frame1.png",
                    "/path/to/frames/video2/frame2.png",
                    ...
                ]
            },
            ...
        },
        "original_sequences": {
            "youtube": {
                "video1": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video1/frame1.png",
                        "/path/to/frames/video1/frame2.png",
                        ...
                    ]
                },
                "video2": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video2/frame1.png",
                        "/path/to/frames/video2/frame2.png",
                        ...
                    ]
                },
                ...
            }
        }
    }
}
"""


import os
import glob
import re
import cv2
import json
import yaml
import pandas as pd
from pathlib import Path


def generate_dataset_file(dataset_name, dataset_root_path, output_file_path, compression_level='c23', perturbation = 'end_to_end'):
    """
    Description:
        - Generate a JSON file containing information about the specified datasets' videos and frames.
    Args:
        - dataset: The name of the dataset.
        - dataset_path: The path to the dataset.
        - output_file_path: The path to the output JSON file.
        - compression_level: The compression level of the dataset.
    """

    # Initialize an empty dictionary to store dataset information.
    dataset_dict = {}


    ## FaceForensics++ dataset or DeepfakeDetection dataset
    ## Note: DeepfakeDetection dataset is a subset of FaceForensics++ dataset
    if dataset_name == 'FaceForensics++' or dataset_name == 'DeepFakeDetection' or dataset_name == 'FaceShifter': 
        ff_dict = {
            'Deepfakes': 'FF-DF',
            'Face2Face': 'FF-F2F',
            'FaceSwap': 'FF-FS',
            'Real': 'FF-real',
            'DFD_Real': 'DFD_real',
            'NeuralTextures': 'FF-NT',
            'FaceShifter': 'FF-FH',
            'DeepFakeDetection': 'DFD_fake',
            'DeepFakeDetection_original': 'DFD_real',
        }
        # Load the JSON files for data split
        dataset_path = os.path.join(dataset_root_path, 'FaceForensics++')
        
        # Load the JSON files for data split
        with open(file=os.path.join(os.path.join(dataset_root_path, 'FaceForensics++', 'train.json')), mode='r') as f:
            train_json = json.load(f)
        with open(file=os.path.join(os.path.join(dataset_root_path, 'FaceForensics++', 'val.json')), mode='r') as f:
            val_json = json.load(f)
        with open(file=os.path.join(os.path.join(dataset_root_path, 'FaceForensics++', 'test.json')), mode='r') as f:
            test_json = json.load(f)
            
        # Create a dictionary for searching the data split 
        video_to_mode = dict()
        for d1, d2 in train_json:
            video_to_mode[d1] = 'train'
            video_to_mode[d2] = 'train'
            video_to_mode[d1+'_'+d2] = 'train'
            video_to_mode[d2+'_'+d1] = 'train'
        for d1, d2 in val_json:
            video_to_mode[d1] = 'val'
            video_to_mode[d2] = 'val'
            video_to_mode[d1+'_'+d2] = 'val'
            video_to_mode[d2+'_'+d1] = 'val'
        for d1, d2 in test_json:
            video_to_mode[d1] = 'test'
            video_to_mode[d2] = 'test'
            video_to_mode[d1+'_'+d2] = 'test'
            video_to_mode[d2+'_'+d1] = 'test'
        
        
        # FaceForensics++ real dataset
        if os.path.isdir(dataset_path) and os.path.isdir(os.path.join(dataset_path, 'original_sequences')):
            label = 'Real'
            dataset_dict['FaceForensics++'] = {}
            dataset_dict['FaceForensics++']['FF-real'] = {}
            dataset_dict['FaceForensics++']['DFD_real'] = {}
            
            # Iterate over all compression levels: c23, c40, raw
            dataset_dict['FaceForensics++']['FF-real']['train'] = {}
            dataset_dict['FaceForensics++']['FF-real']['test'] = {}
            dataset_dict['FaceForensics++']['FF-real']['val'] = {}
            for compression_level in os.scandir(os.path.join(dataset_path, 'original_sequences', 'youtube')):
                frame_folder_path = os.path.join(dataset_path, 'original_sequences', 'youtube', compression_level, 'frames_all')
                if not os.path.exists(frame_folder_path):
                    print(f"Folder not exists: {frame_folder_path}, skip it")
                    continue
                if compression_level.is_dir():
                    compression_level = compression_level.name
                    dataset_dict['FaceForensics++']['FF-real']['train'][compression_level] = {}
                    dataset_dict['FaceForensics++']['FF-real']['test'][compression_level] = {}
                    dataset_dict['FaceForensics++']['FF-real']['val'][compression_level] = {}
            
                # Iterate over all videos
                for video_path in os.scandir(frame_folder_path):
                    if video_path.is_dir():
                        video_name = video_path.name
                        mode = video_to_mode[video_name]
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict['FaceForensics++']['FF-real'][mode][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
                        
            label = 'DFD_Real'  
            # Same operations for DeepfakeDetection real dataset
            dataset_dict['FaceForensics++']['DFD_real']['train'] = {}
            dataset_dict['FaceForensics++']['DFD_real']['test'] = {}
            dataset_dict['FaceForensics++']['DFD_real']['val'] = {}
            for compression_level in os.scandir(os.path.join(dataset_path, 'original_sequences', 'actors')):
                frame_folder_path = os.path.join(dataset_path, 'original_sequences', 'actors', compression_level, 'frames')
                if not os.path.exists(frame_folder_path):
                    print(f"Folder not exists: {frame_folder_path}, skip it")
                    continue
                if compression_level.is_dir() and compression_level.name in ["c23", "c40", "raw"]:
                    compression_level = compression_level.name
                    dataset_dict['FaceForensics++']['DFD_real']['train'][compression_level] = {}
                    dataset_dict['FaceForensics++']['DFD_real']['test'][compression_level] = {}
                    dataset_dict['FaceForensics++']['DFD_real']['val'][compression_level] = {}
                # Iterate over all videos
                for video_path in os.scandir(frame_folder_path):
                    if video_path.is_dir():
                        video_name = video_path.name
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict['FaceForensics++']['DFD_real']['train'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
                        dataset_dict['FaceForensics++']['DFD_real']['test'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
                        dataset_dict['FaceForensics++']['DFD_real']['val'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
        # FaceForensics++ fake datasets
        if os.path.isdir(os.path.join(dataset_path, 'manipulated_sequences')):
            for label_dir in os.scandir(os.path.join(dataset_path, 'manipulated_sequences')):
                if label_dir.is_dir():
                    label = label_dir.name
                    dataset_dict['FaceForensics++'][ff_dict[label]] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['train'] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['test'] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['val'] = {}
                    
                    # Iterate over all compression levels: c23, c40, raw
                    for compression_level in os.scandir(os.path.join(dataset_path, 'manipulated_sequences', label)):
                        frame_folder_path = os.path.join(dataset_path, 'manipulated_sequences', label, compression_level, 'frames_all')
                        if not os.path.exists(frame_folder_path):
                            print(f"Folder not exists: {frame_folder_path}, skip it")
                            continue
                        if compression_level.is_dir() and compression_level.name in ["c23", "c40", "raw"]:
                            compression_level = compression_level.name
                            dataset_dict['FaceForensics++'][ff_dict[label]]['train'][compression_level] = {}
                            dataset_dict['FaceForensics++'][ff_dict[label]]['test'][compression_level] = {}
                            dataset_dict['FaceForensics++'][ff_dict[label]]['val'][compression_level] = {}
                            # Iterate over all videos

                            for video_path in os.scandir(frame_folder_path):
                                if video_path.is_dir():
                                    video_name = video_path.name
                                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                                    if label != 'FaceShifter':
                                        mask_paths = os.path.join(dataset_path, 'manipulated_sequences', label, 'c23','masks', video_name)
                                        # mask is all the same for all compression levels
                                        if os.path.exists(mask_paths):
                                            mask_frames_paths = [os.path.join(mask_paths, frame.name) for frame in os.scandir(mask_paths)]
                                        else:
                                            mask_frames_paths = []
                                        try:
                                            mode = video_to_mode[video_name]
                                            dataset_dict['FaceForensics++'][ff_dict[label]][mode][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                        # DeepfakeDetection dataset
                                        except:
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['train'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['val'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['test'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                    # FaceShifter dataset
                                    else:
                                        mode = video_to_mode[video_name]
                                        dataset_dict['FaceForensics++'][ff_dict[label]][mode][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
         

        # get the DeepfakeDetection dataset from FaceForensics++ dataset
        if dataset_name == 'FaceForensics++':
            # Delete the DeepfakeDetection dataset from FaceForensics++ dataset
            del dataset_dict['FaceForensics++']['DFD_fake']
            del dataset_dict['FaceForensics++']['DFD_real']
            del dataset_dict['FaceForensics++']['FF-FH']
        elif dataset_name == 'DeepFakeDetection':
            # Check if the DeepfakeDetection dataset is in the FaceForensics++ dataset
            if 'DFD_fake' in dataset_dict['FaceForensics++'] and \
                'DFD_real' in dataset_dict['FaceForensics++']:
                # Add the DeepfakeDetection dataset to the dataset_dict
                dataset_dict['DeepFakeDetection'] = {
                    'DFD_fake': dataset_dict['FaceForensics++']['DFD_fake'], 
                    'DFD_real': dataset_dict['FaceForensics++']['DFD_real']
                }
                del dataset_dict['FaceForensics++']
        elif dataset_name == 'FaceShifter':
            if 'FF-FH' in dataset_dict['FaceForensics++'] and \
                'FF-real' in dataset_dict['FaceForensics++']:
                # Add the DeepfakeDetection dataset to the dataset_dict
                dataset_dict['FaceShifter'] = {
                    'FF-FH': dataset_dict['FaceForensics++']['FF-FH'], 
                    'FF-real': dataset_dict['FaceForensics++']['FF-real']
                }
                del dataset_dict['FaceForensics++']
            else:
                # TODO
                raise ValueError('DeepfakeDetection dataset not found in FaceForensics++ dataset.')
        else:
            raise ValueError('Invalid dataset name: {}'.format(dataset_name))

        # if FaceForensics++, based on label and generate the json
        if dataset_name == 'FaceForensics++':
            for label, value in dataset_dict['FaceForensics++'].items():
                if label != 'FF-real':
                    with open(os.path.join(output_file_path,f'{label}.json'), 'w') as f:
                        data = {label: {'FF-real': dataset_dict['FaceForensics++']['FF-real'],
                                        label: value,
                                        }}
                        json.dump(data, f)
                        print(f"Finish writing {label}.json")
    
    ## Celeb-DF-v1 dataset
    ## Note: videos in Celeb-DF-v1/2 are not in the same format as in FaceForensics++ dataset
    elif dataset_name == 'Celeb-DF-v1':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv1_real'
            else:
                label = 'CelebDFv1_fake'
            assert label in ['CelebDFv1_real', 'CelebDFv1_fake'], 'Invalid label: {}'.format(label)
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
        
        # Special case for test&val data of Celeb-DF-v1/2
        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv1_real'
            elif 'synthesis' in line:
                label = 'CelebDFv1_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    ## Celeb-DF-v2 dataset
    ## Note: videos in Celeb-DF-v1/2 are not in the same format as in FaceForensics++ dataset
    elif dataset_name == 'Celeb-DF-v2':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv2_real'
            else:
                label = 'CelebDFv2_fake'
            assert label in ['CelebDFv2_real', 'CelebDFv2_fake'], 'Invalid label: {}'.format(label)
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
        
        # Special case for test&val data of Celeb-DF-v1/2
        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv2_real'
            elif 'synthesis' in line:
                label = 'CelebDFv2_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    ## DFDCP dataset
    elif dataset_name == 'DFDCP':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        #initialize the dataset dictionary
        dataset_dict[dataset_name] = {'DFDCP_Real': {'train': {}, 'test': {}, 'val': {}},
                                'DFDCP_FakeA': {'train': {}, 'test': {}, 'val': {}},
                                'DFDCP_FakeB': {'train': {}, 'test': {}, 'val': {}}}
        # Open the dataset information file ('dataset.json') and parse its contents
        with open(os.path.join(dataset_path, 'dataset.json' ), 'r') as f:
            dataset_info = json.load(f)
        # Iterate over the dataset_info dictionary and extract the index and file name for each video
        for dataset in dataset_info.keys():
            index = dataset.split('/')[0]
            vidname = dataset.split('/')[-1].split(".")[0]
            if Path(os.path.join(dataset_path, index, 'frames', vidname)).exists():
                frame_paths = glob.glob(os.path.join(dataset_path, index, 'frames', vidname, '*png'))
                if len(frame_paths) == 0:
                    continue
                label = dataset_info[dataset]['label']
                if label == 'real':
                    label = 'DFDCP_Real'
                elif label == 'fake' and index == 'method_A':
                    label = 'DFDCP_FakeA'
                elif label == 'fake' and index == 'method_B':
                    label = 'DFDCP_FakeB'
                else:
                    raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
                set_attr = dataset_info[dataset]['set']  # train, test, val
                dataset_dict[dataset_name][label][set_attr][vidname] = {'label': label, 'frames': frame_paths}
        # Special case for val data of DFDCP
        for label in ['DFDCP_Real', 'DFDCP_FakeA', 'DFDCP_FakeB']:
            dataset_dict[dataset_name][label]['val'] = dataset_dict[dataset_name][label]['test']
    
    ## DFDC dataset
    elif dataset_name == 'DFDC':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {'DFDC_Real': {'train': {}, 'test': {}, 'val': {}},
                                'DFDC_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['test']:
                # 读取csv文件
                df = pd.read_csv(os.path.join(dataset_path,folder.name,'labels.csv'))
                labels = ['DFDC_Real','DFDC_Fake']
                # 循环遍历每一行，并逐行读取filename和label的值
                for index, row in df.iterrows():
                    vidname = row['filename'].split('.mp4')[0]
                    label = labels[row['label']]
                    assert label in ['DFDC_Real','DFDC_Fake'], 'Invalid label: {}'.format(label)
                    frame_paths = glob.glob(os.path.join(dataset_path, folder.name,'frames', vidname, '*png'))
                    if len(frame_paths) == 0:
                        continue
                    dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
                    dataset_dict[dataset_name][label]['val'] = {'label': label, 'frames': frame_paths}
            
            elif folder.name in ['train']:
                num_file = 0
                for dfdc_train_part in os.scandir(os.path.join(dataset_path, folder.name)):
                    if not os.path.isdir(dfdc_train_part):
                        continue
                    num_file += 1
                    print('processing {}th file in 50 files.'.format(num_file))
                    with open(os.path.join(dfdc_train_part, 'metadata.json'), 'r') as f:
                            metadata = json.load(f)
                    for video_path in os.scandir(os.path.join(dfdc_train_part, 'frames')):
                        if video_path.is_dir():
                            video_name = video_path.name
                            label = metadata[video_name + ".mp4"]["label"]
                            assert label in ['REAL', 'FAKE'], 'Invalid label: {}'.format(label)
                            if label == 'REAL':
                                label = 'DFDC_Real'
                            else:
                                label = 'DFDC_Fake'
                            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                            dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
        
    ## UADFV dataset
    elif dataset_name == 'UADFV':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {'UADFV_Real': {'train': {}, 'test': {}, 'val': {}},
                                'UADFV_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            elif folder.name in ['fake']:
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Fake'
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
            elif folder.name in ['real']:
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Real'
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

    ## roop dataset
    elif dataset_name == 'roop':
        dataset_path = os.path.join(dataset_root_path, dataset_name, 'cdfv2', 'faces_all')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'roop_Real'
            elif 'synthesis' in line:
                continue
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}


    
    elif dataset_name == 'roop_ff':
        dataset_path = os.path.join(dataset_root_path, 'roop', 'ffpp', 'faces_all')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'test.json'), 'r') as fd:
            data = json.load(fd)
        videos = [os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in data for item in sublist]
        for video_path in videos:
            video_name = video_path.split('/')[-1]
            label = 'roop_Real'
            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

            



    


    ## uniface dataset
    elif dataset_name == 'uniface':
        dataset_path = os.path.join(dataset_root_path, dataset_name, 'cdfv2', 'frames')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'roop_Real'
            elif 'synthesis' in line:
                continue
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}


    
    elif dataset_name == 'uniface_ff':
        dataset_path = os.path.join(dataset_root_path, 'uniface', 'ffpp', 'frames')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'test.json'), 'r') as fd:
            data = json.load(fd)
        videos = [os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in data for item in sublist]
        for video_path in videos:
            video_name = video_path.split('/')[-1]
            label = 'roop_Real'
            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}










    ## roop dataset
    elif dataset_name == 'simswap':
        dataset_path = os.path.join(dataset_root_path, dataset_name, 'cdfv2', 'frames')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'roop_Real'
            elif 'synthesis' in line:
                continue
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}


    
    elif dataset_name == 'simswap_ff':
        dataset_path = os.path.join(dataset_root_path, 'simswap', 'ffpp', 'frames')
        dataset_dict[dataset_name] = {'roop_Real': {'train': {}, 'test': {}, 'val': {}},
                                'roop_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'roop_Fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

        
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'test.json'), 'r') as fd:
            data = json.load(fd)
        videos = [os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in data for item in sublist]
        for video_path in videos:
            video_name = video_path.split('/')[-1]
            label = 'roop_Real'
            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}




    elif dataset_name == 'DeeperForensics-1.0':
        dataset_path = '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/DeeperForensics-1.0/end_to_end/frames'
        dataset_dict[dataset_name] = {'DF_real': {'train': {}, 'test': {}, 'val': {}},
                                'DF_fake': {'train': {}, 'test': {}, 'val': {}}}
        for video_path in os.scandir(dataset_path):
            if video_path.is_dir():
                video_name = video_path.name
                label = 'DF_fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
        
        with open(os.path.join('/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'test.json'), 'r') as fd:
            data = json.load(fd)
        videos = [os.path.join('/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in data for item in sublist]
        for video_path in videos:
            video_name = video_path.split('/')[-1]
            label = 'DF_real'
            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}



    elif dataset_name == 'e4e_ff':
        dataset_path = '/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/DF40/e4e'
        dataset_dict[dataset_name] = {'e4e_Real': {'train': {}, 'test': {}, 'val': {}},
                                'e4e_Fake': {'train': {}, 'test': {}, 'val': {}}}
        # ff domain for training and validation
        for video_path in os.scandir(os.path.join(dataset_path, 'ff', 'inversions')):                                
            video_name = video_path.name.split('.')[0]
            frame_name = video_path.name
            label = 'e4e_Fake'
            frame_paths = [os.path.join(video_path)]
            dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
        

        # real data for training
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'train.json'), 'r') as fd:
            train_data = json.load(fd)
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++', 'test.json'), 'r') as fd:
            test_data = json.load(fd)
        train_videos = [os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in train_data for item in sublist]
        test_videos = [os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames', item) for sublist in test_data for item in sublist]
        for video_path in train_videos:
            try:
                video_name = video_path.split('/')[-1]
                label = 'e4e_Real'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
            except Exception as e:
                print(e)
        for video_path in test_videos:
            video_name = video_path.split('/')[-1]
            label = 'e4e_Real'
            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}

    

    elif dataset_name == 'e4e_cdf':
        dataset_path = '/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/DF40/e4e'
        dataset_dict[dataset_name] = {'e4e_Real': {'train': {}, 'test': {}, 'val': {}},
                                'e4e_Fake': {'train': {}, 'test': {}, 'val': {}}}
        # cdf domain for testing
        for video_path in os.scandir(os.path.join(dataset_path, 'cdf', 'Celeb-real', 'inversions')):
            video_name = video_path.name.split('.')[0]
            frame_name = video_path.name
            label = 'e4e_Fake'
            frame_paths = [os.path.join(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
        for video_path in os.scandir(os.path.join(dataset_path, 'cdf', 'YouTube-real', 'inversions')):
            video_name = video_path.name.split('.')[0]
            frame_name = video_path.name
            label = 'e4e_Fake'
            frame_paths = [os.path.join(video_path)]
            dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths} 
        

        # real data for test
        with open(os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'e4e_Real'
            elif 'synthesis' in line:
                continue
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join('/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/Celeb-DF-v2', line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}




    # Convert the dataset dictionary to JSON format and save to file
    output_file_path = os.path.join(output_file_path, dataset_name + '.json')
    with open(output_file_path, 'w') as f:
        json.dump(dataset_dict, f)
    # print the successfully generated dataset dictionary
    print(f"{dataset_name}.json generated successfully.")

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']
    # Call the generate_dataset_file function
    generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)
