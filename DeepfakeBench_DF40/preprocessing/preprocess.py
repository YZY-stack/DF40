# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""
Original dataset structure before the preprocessing:

-FaceForensics++
    -original_sequences
        -youtube
            -c23
                -videos
                    *.mp4
    -manipulated_sequences
        -Deepfakes
            -c23
                -videos
        -Face2Face
            -c23
                -videos
        -FaceSwap
            -c23
                -videos
        -NeuralTextures
            -c23
                -videos
        -FaceShifter
            -c23
                -videos
        -DeepFakeDetection
            -c23
                -videos
                
-Celeb-DF-v1/v2
    -Celeb-synthesis
        -videos
    -Celeb-real
        -videos
    -YouTube-real
        -videos

-DFDCP
    -method_A
    -method_B
    -original_videos

-DeeperForensics-1.0
    -manipulated_videos
    -source_videos

We then additionally obtain "frames", "landmarks", and "mask" directories in same directory as the "videos" folder.
"""


import os
import sys
import time
import cv2
import dlib
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans


def create_logger(log_path):
    """
    Creates a logger object and saves all messages to a file.

    Args:
        log_path (str): The path to save the log file.

    Returns:
        logger: The logger object.
    """
    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # Calculate the center of the image
        img_center = np.array([rgb.shape[1] // 2, rgb.shape[0] // 2])

        # For now only take the biggest face and the face closest to the center
        face = max(faces, 
                   key=lambda rect: rect.width() * rect.height() and np.linalg.norm(np.array([(rect.left() + rect.right()) // 2, 
                   (rect.top() + rect.bottom()) // 2]) - img_center)
        )

        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    
    else:
        return None, None, None

def video_manipulate(
    movie_path: Path,
    mask_path: Path,
    dataset_path: Path,
    mode: str,
    num_frames: int, 
    stride: int, 
    ) -> None:
    """
    Processes a single video file by detecting and cropping the largest face in each frame and saving the results.

    Args:
        movie_path (str): Path to the video file to process.
        dataset_path (str): Path to the dataset directory.
        mask_path (str): Path to the mask directory.
        mode (str): Either 'fixed_num_frames' or 'fixed_stride'.
        num_frames (int): Number of frames to extract from the video.
        stride (int): Number of frames to skip between each frame extracted.
        margin (float): Amount to increase the size of the face bounding box by.
        visualization (bool): Whether to save visualization images.

    Returns:
        None
    """

    # Define face detector and predictor models
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        logger.error(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)
    
    def facecrop(
        org_path: Path,
        mask_path: Path, 
        save_path: Path, 
        mode: str,
        num_frames: int, 
        stride: int,
        face_predictor: dlib.shape_predictor, 
        face_detector: dlib.fhog_object_detector,
        margin: float = 0.5, 
        visualization: bool = False
        ) -> None:
        """
        Helper function for cropping face and extracting landmarks.
        """
        
        # Open the video file
        assert org_path.exists(), f"Video file {org_path} does not exist."
        cap_org = cv2.VideoCapture(str(org_path))
        if not cap_org.isOpened():
            logger.error(f"Failed to open {org_path}")
            return

        if mask_path is not None:
            cap_mask = cv2.VideoCapture(str(mask_path))
            if not cap_mask.isOpened():
                logger.error(f"Failed to open {mask_path}")
                return
        
        # Get the number of frames in the video
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the mode
        if mode == 'fixed_num_frames':
            # Get the frame rate of the video by dividing the number of frames by the duration (same interval between frames)
            frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
        elif mode == 'fixed_stride':
            # Get the frame rate of the video by dividing the number of frames by the duration (same interval between frames)
            frame_idxs = np.arange(0, frame_count_org, stride, dtype=int)

        # Iterate through the frames
        for cnt_frame in range(frame_count_org):
            ret_org, frame_org = cap_org.read()
            if mask_path is not None:
                ret_mask, frame_mask = cap_mask.read()
            else:
                frame_mask = None
            height, width = frame_org.shape[:-1]

            # Save original extracted frames
            frame_path__ = save_path / 'frames_wocropface' / org_path.stem
            frame_path__.mkdir(parents=True, exist_ok=True)
            # Save
            ori_frame_path = frame_path__ / f"{cnt_frame:03d}.png"
            if not ori_frame_path.is_file():
                cv2.imwrite(str(ori_frame_path), frame_org)

            # Check if the frame was successfully read
            if not ret_org:
                logger.warning(f"Failed to read frame {cnt_frame} of {org_path}")
                break
            
            # Check if the mask was successfully read
            if mask_path is not None and not ret_mask:
                logger.warning(f"Failed to read mask {cnt_frame} of {mask_path}")
                break
            # Check if the frame is one of the frames to extract
            if cnt_frame not in frame_idxs:
                continue

            # Use the function to extract the aligned and cropped face
            if mask_path is not None:
                cropped_face, landmarks, masks = extract_aligned_face_dlib(face_detector, face_predictor, frame_org, mask=frame_mask)
            else:
                cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, face_predictor, frame_org, mask=frame_mask)
            
            # Check if a face was detected and cropped
            if cropped_face is None:
                logger.warning(f"No faces in frame {cnt_frame} of {org_path}")
                continue
            
            # Check if the landmarks were detected
            if landmarks is None:
                logger.warning(f"No landmarks in frame {cnt_frame} of {org_path}")
                continue

            # Save cropped face, landmarks, and visualization image
            save_path_ = save_path / 'frames' / org_path.stem
            save_path_.mkdir(parents=True, exist_ok=True)

            # Save cropped face
            image_path = save_path_ / f"{cnt_frame:03d}.png"
            if not image_path.is_file():
                cv2.imwrite(str(image_path), cropped_face)

            # Save landmarks
            land_path = save_path / 'landmarks' / org_path.stem / f"{cnt_frame:03d}.npy"
            os.makedirs(os.path.dirname(land_path), exist_ok=True)
            np.save(str(land_path), landmarks)

            # Save mask
            if mask_path is not None:
                mask_path = save_path / 'masks' / org_path.stem / f"{cnt_frame:03d}.png"
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                _, binary_mask = cv2.threshold(masks, 1, 255, cv2.THRESH_BINARY)  # obtain binary mask only
                cv2.imwrite(str(mask_path), binary_mask)

        # Release the video capture
        cap_org.release()
        if mask_path is not None:
            cap_mask.release()

    # Iterate through the videos in the dataset and extract faces
    try:
        facecrop(movie_path, mask_path, dataset_path, mode, num_frames, stride, face_predictor, face_detector)
    except Exception as e:
        logger.error(f"Error processing video {movie_path}: {e}")


def preprocess(dataset_path, mask_path, mode, num_frames, stride, logger):
    # Define paths to videos in dataset
    movies_path_list = sorted([Path(p) for p in glob.glob(os.path.join(dataset_path, '**/*.mp4'), recursive=True)])
    if len(movies_path_list) == 0:
        logger.error(f"No videos found in {dataset_path}")
        sys.exit()
    logger.info(f"{len(movies_path_list)} videos found in {dataset_path}")
    
    # Define paths to masks in dataset
    if mask_path is not None:
        masks_path_list = sorted([Path(p) for p in glob.glob(os.path.join(mask_path, '**/*.mp4'), recursive=True)])
        if len(masks_path_list) == 0:
            logger.error(f"No masks found in {mask_path}")
            # sys.exit()
        logger.info(f"{len(masks_path_list)} masks found in {mask_path}")    
    
    # Start timer
    start_time = time.monotonic()

    # Define the number of processes based on CPU capabilities
    num_processes = os.cpu_count()

    # Use multiprocessing to process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for movie_path in movies_path_list:
            # Check if there is a mask for the video
            if mask_path is not None:
                if movie_path.stem not in [path.stem for path in masks_path_list]:
                    logger.error(f"No mask for video {movie_path}")
                # Define the mask path
                mask_path = next((path for path in masks_path_list if path.stem == movie_path.stem), None)
                if mask_path is None:
                    logger.error(f"Mask path not found for video {movie_path}")
            # Create a future for each video and submit it for processing
            futures.append(
                executor.submit(
                video_manipulate,
                movie_path,
                mask_path,
                dataset_path,
                mode,
                num_frames,
                stride,
                )
            )
        # Wait for all futures to complete and log any errors
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(movies_path_list)):
            # Print the current time
            logger.info(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")
            
        # End timer
        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        logger.info(f"Total time taken: {duration_minutes:.2f} minutes")

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './preprocessing/config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    # Get the parameters
    dataset_name = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    comp = config['preprocess']['comp']['default']
    mode = config['preprocess']['mode']['default']
    stride = config['preprocess']['stride']['default']
    num_frames = config['preprocess']['num_frames']['default']
    
    # use dataset_name and dataset_root_path to get dataset_path
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))


    # Create logger
    log_path = f'./{dataset_name}.log'
    logger = create_logger(log_path)

    # Define dataset path based on the input arguments
    ## faceforensic++
    if dataset_name == 'FaceForensics++':
        sub_dataset_names = ["original_sequences/youtube","original_sequences/actors", \
                             "manipulated_sequences/Deepfakes", \
                            "manipulated_sequences/Face2Face", "manipulated_sequences/FaceSwap", \
                            "manipulated_sequences/NeuralTextures","manipulated_sequences/FaceShifter",\
                            "manipulated_sequences/DeepFakeDetection"]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, comp)) for name in sub_dataset_names]
        # mask
        mask_dataset_names = ["manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face", \
                            "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures",\
                            "manipulated_sequences/DeepFakeDetection"]
        # mask_dataset_names = []
        mask_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in mask_dataset_names]
    ## Celeb-DF-v1
    elif dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    ## Celeb-DF-v2
    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    ## DFDCP
    elif dataset_name == 'DFDCP':
        sub_dataset_names = ['original_videos', 'method_A', 'method_B']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## DFDC-test
    elif dataset_name == 'DFDC':  # only consider test set for DFDC
        sub_dataset_names = ['test']
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'test'))]
   
    ## DeeperForensics-1.0
    elif dataset_name == 'DeeperForensics-1.0':
        fake_sub_dataset_names = ['end_to_end/' + name for name in os.listdir(os.path.join(dataset_path, 'end_to_end'))]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in fake_sub_dataset_names]
        
    ## UADFV
    elif dataset_name == 'UADFV':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    # DF40
    elif dataset_name == 'DF40_cdf':
        dataset_path = Path(str(dataset_path).replace('DF40_cdf', 'DF40'))
        aigc_dataset_name = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'ddim', 'ddpm', 'collaborative_diffusion', 'pixart', 'SiT', 'sd1.5', 'sd2.1', 'VQGAN', 'DiT', 'MidJourney', 'dalle2_face']
        # obtain all forgery names within df40
        sub_dataset_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d not in aigc_dataset_name]
        # obtain all video names for each forgery folder
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, 'cdf')) for name in sub_dataset_names]
    elif dataset_name == 'DF40_ff':
        dataset_path = Path(str(dataset_path).replace('DF40_ff', 'DF40'))
        aigc_dataset_name = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'ddim', 'ddpm', 'collaborative_diffusion', 'pixart', 'SiT', 'sd1.5', 'sd2.1', 'VQGAN', 'DiT', 'MidJourney', 'dalle2_face' ]
        # obtain all forgery names within df40
        sub_dataset_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d not in aigc_dataset_name]
        # obtain all video names for each forgery folder
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, 'ff')) for name in sub_dataset_names]
    
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    
    # Check if dataset path exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit()

    if 'sub_dataset_paths' in globals() and len(sub_dataset_paths) != 0:
        # Check if sub_dataset path exists
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset path does not exist: {sub_dataset_path}")
                sys.exit()
        # preprocess each sub_dataset
        for sub_dataset_path in sub_dataset_paths:
            # only part of FaceForensics++ has mask
            if dataset_name == 'FaceForensics++' and sub_dataset_path.parent in mask_dataset_paths:
                mask_dataset_path = os.path.join(sub_dataset_path.parent, "masks")
                preprocess(sub_dataset_path, mask_dataset_path, mode, num_frames, stride, logger)
            else:
                preprocess(sub_dataset_path, None, mode, num_frames, stride, logger)
    else:
        logger.error(f"Sub Dataset path does not exist: {sub_dataset_paths}")
        sys.exit()
    logger.info("Face cropping complete!")
