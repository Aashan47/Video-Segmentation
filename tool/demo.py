import os
import time
import logging
import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from time import perf_counter
from collections import defaultdict

# Adjust import paths to include the parent directory
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from util import dataset, transform, config
from util.util import AverageMeter, check_makedirs, transfer_ckpt, colorize
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert
from scipy.spatial import KDTree  # For faster nearest neighbor search
from scipy.optimize import linear_sum_assignment  # For Hungarian matching

import cv2
import numpy as np
import logging
from collections import defaultdict
from scipy.spatial.distance import cdist
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

class SpatialTracker:
    """
    Tracks spatial regions to identify when the camera revisits previously seen areas.
    Uses feature matching and homography estimation to align frames and create a
    spatial map of previously seen regions.
    """
    
    def __init__(self, 
                 feature_threshold=500,
                 match_threshold=0.75, 
                 ransac_threshold=5.0,
                 revisit_threshold=100,
                 overlap_threshold=0.3,
                 min_inliers=15,
                 debug=True):
        """
        Initialize the spatial tracker.
        
        Args:
            feature_threshold: Minimum feature response threshold for SIFT detector
            match_threshold: Ratio test threshold for feature matching
            ransac_threshold: RANSAC residual threshold for homography estimation
            revisit_threshold: Minimum number of frames between potential revisits
            overlap_threshold: Minimum overlap ratio to consider a frame as revisit
            min_inliers: Minimum number of inliers for a valid homography
            debug: Whether to enable debug logging
        """
        self.feature_detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        self.match_threshold = match_threshold
        self.ransac_threshold = ransac_threshold
        self.revisit_threshold = revisit_threshold
        self.overlap_threshold = overlap_threshold
        self.min_inliers = min_inliers
        
        # Set up storage for keyframes and their features
        self.keyframes = []  # List of keyframe data
        self.keyframe_features = []  # List of (keypoints, descriptors) pairs
        self.revisit_map = np.array([])  # Revisit mask for the current frame
        
        # Store frame-to-frame matches for tracking
        self.matches_history = []
        
        # Set up logger
        self.logger = logging.getLogger('SpatialTracker')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        self.debug = debug
            
    def detect_features(self, frame):
        """
        Detect and compute features for the given frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            keypoints, descriptors
        """
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute SIFT features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match features between two descriptors using ratio test.
        
        Args:
            desc1, desc2: Feature descriptors to match
            
        Returns:
            Filtered matches that pass the ratio test
        """
        # Get top 2 matches for each feature
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    def estimate_transform(self, kp1, kp2, matches):
        """
        Estimate transformation between two sets of keypoints using RANSAC.
        
        Args:
            kp1, kp2: Keypoints from the two frames
            matches: Feature matches between the frames
            
        Returns:
            transformation_matrix, inliers_mask, num_inliers
        """
        if len(matches) < 4:
            return None, None, 0
            
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Use RANSAC to estimate transformation
        try:
            # First try to estimate a similarity transform (less degrees of freedom)
            model, inliers = ransac(
                (src_pts, dst_pts),
                SimilarityTransform, 
                min_samples=3,
                residual_threshold=self.ransac_threshold, 
                max_trials=100
            )
            
            if inliers is None or np.sum(inliers) < self.min_inliers:
                # Fall back to homography if not enough inliers
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
                inliers = mask.ravel().astype(bool)
                
                if np.sum(inliers) < self.min_inliers:
                    return None, None, 0
                    
                return H, inliers, np.sum(inliers)
            
            # Get the similarity transform matrix
            H = model.params
            return H, inliers, np.sum(inliers)
            
        except Exception as e:
            self.logger.warning(f"RANSAC failed: {e}")
            return None, None, 0
    
    def compute_overlap(self, H, frame_shape):
        """
        Compute the overlap between the current frame and a keyframe.
        
        Args:
            H: Homography matrix mapping current frame to keyframe
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            overlap_ratio: Ratio of overlapping area to frame area
        """
        height, width = frame_shape[:2]
        
        # Define corners of the current frame
        corners = np.float32([
            [0, 0],
            [0, height-1],
            [width-1, height-1],
            [width-1, 0]
        ]).reshape(-1, 1, 2)
        
        # Transform corners to keyframe coordinates
        try:
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Create masks for both frames
            mask_current = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_current, [np.int32(corners)], 1)
            
            mask_transformed = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_transformed, [np.int32(transformed_corners)], 1)
            
            # Calculate overlap
            intersection = np.logical_and(mask_current, mask_transformed).sum()
            union = np.logical_or(mask_current, mask_transformed).sum()
            
            if union == 0:
                return 0.0
                
            overlap_ratio = intersection / float(union)
            return overlap_ratio
            
        except Exception as e:
            self.logger.warning(f"Error computing overlap: {e}")
            return 0.0
    
    def is_keyframe(self, frame, frame_idx, kp, desc):
        """
        Determine if the current frame should be stored as a keyframe.
        
        Args:
            frame: Current frame
            frame_idx: Frame index
            kp: Keypoints detected in the current frame
            desc: Descriptors for the keypoints
            
        Returns:
            is_keyframe (bool): Whether this frame should be a keyframe
        """
        # Always add the first frame as a keyframe
        if len(self.keyframes) == 0:
            return True
            
        # If not enough features, don't make it a keyframe
        if len(kp) < 100:
            return False
            
        # Check against the most recent keyframe
        last_kp, last_desc = self.keyframe_features[-1]
        
        # Match features
        matches = self.match_features(last_desc, desc)
        
        # If not enough matches, this is a new scene - make it a keyframe
        if len(matches) < 30:
            return True
            
        # Estimate transform
        H, inliers, num_inliers = self.estimate_transform(last_kp, kp, matches)
        
        # If transform estimation failed, make it a keyframe
        if H is None:
            return True
            
        # Calculate overlap
        overlap = self.compute_overlap(H, frame.shape)
        
        # If overlap is small, this is a new view - make it a keyframe
        if overlap < 0.6:  # Less than 60% overlap with previous keyframe
            return True
            
        return False
    
    def detect_revisit(self, frame, frame_idx, kp, desc):
        """
        Detect if the current frame revisits a previously seen location.
        
        Args:
            frame: Current frame
            frame_idx: Frame index
            kp: Keypoints detected in the current frame
            desc: Descriptors for the keypoints
            
        Returns:
            revisit_info: Dictionary with revisit information or None if no revisit
        """
        if len(self.keyframes) < 5:  # Need some keyframes to detect revisits
            return None
            
        # Skip recent keyframes (can't revisit very recent frames)
        search_keyframes = []
        for i, (kf_idx, _, _) in enumerate(self.keyframes[:-2]):  # Skip the last 2 keyframes
            if frame_idx - kf_idx > self.revisit_threshold:
                search_keyframes.append(i)
                
        if not search_keyframes:
            return None
            
        best_match = None
        best_inliers = 0
        best_overlap = 0
        best_keyframe_idx = -1
        
        # Test against all candidate keyframes
        for i in search_keyframes:
            kf_idx, _, _ = self.keyframes[i]
            kf_kp, kf_desc = self.keyframe_features[i]
            
            # Match features
            matches = self.match_features(kf_desc, desc)
            
            # Need enough matches for a reliable transform
            if len(matches) < self.min_inliers:
                continue
                
            # Estimate transform
            H, inliers, num_inliers = self.estimate_transform(kf_kp, kp, matches)
            
            if H is None or num_inliers < self.min_inliers:
                continue
                
            # Calculate overlap
            overlap = self.compute_overlap(H, frame.shape)
            
            # Check if this is the best match so far
            # We prioritize overlap, but also consider the number of inliers
            score = overlap * (num_inliers / 100.0)
            if overlap > self.overlap_threshold and (
                best_match is None or score > best_overlap * (best_inliers / 100.0)
            ):
                best_match = H
                best_inliers = num_inliers
                best_overlap = overlap
                best_keyframe_idx = i
                
        # If we found a good match
        if best_match is not None and best_overlap > self.overlap_threshold:
            kf_idx, kf_frame, kf_mask = self.keyframes[best_keyframe_idx]
            
            # Create revisit mask by transforming the keyframe mask
            height, width = frame.shape[:2]
            revisit_mask = cv2.warpPerspective(
                kf_mask, 
                best_match, 
                (width, height), 
                flags=cv2.INTER_NEAREST
            )
            
            # Update the revisit map
            self.revisit_map = revisit_mask > 0
            
            return {
                'keyframe_idx': kf_idx,
                'current_idx': frame_idx,
                'transform': best_match,
                'overlap': best_overlap,
                'inliers': best_inliers
            }
        
        # No revisit detected
        self.revisit_map = np.zeros(frame.shape[:2], dtype=bool)
        return None
    
    def update(self, frame, frame_idx, label_mask=None):
        """
        Process a new frame and update the spatial map.
        
        Args:
            frame: Current video frame (BGR format)
            frame_idx: Frame index
            label_mask: Optional mask of labeled regions
            
        Returns:
            revisit_info: Information about revisited region or None
            revisit_mask: Binary mask of revisited regions
        """
        # Initialize masks
        height, width = frame.shape[:2]
        if label_mask is None:
            label_mask = np.ones((height, width), dtype=np.uint8)
            
        # Extract features from current frame
        kp, desc = self.detect_features(frame)
        
        self.logger.debug(f"Frame {frame_idx}: Detected {len(kp)} keypoints")
        
        # Check for revisits
        revisit_info = self.detect_revisit(frame, frame_idx, kp, desc)
        
        if revisit_info:
            self.logger.info(f"Frame {frame_idx}: Revisiting frame {revisit_info['keyframe_idx']} "
                           f"with overlap {revisit_info['overlap']:.2f}, inliers {revisit_info['inliers']}")
            
        # Check if this should be a new keyframe
        if self.is_keyframe(frame, frame_idx, kp, desc):
            self.logger.debug(f"Frame {frame_idx}: Adding as keyframe #{len(self.keyframes)}")
            
            # Store keyframe with its mask of labeled regions
            self.keyframes.append((frame_idx, frame.copy(), label_mask.copy()))
            self.keyframe_features.append((kp, desc))
            
        return revisit_info, self.revisit_map
    
    def get_revisit_mask(self):
        """
        Get the mask of revisited regions for the current frame.
        
        Returns:
            Binary mask of revisited regions
        """
        return self.revisit_map
    
    def visualize(self, frame, revisit_info=None):
        """
        Visualize the spatial tracking results.
        
        Args:
            frame: Current video frame
            revisit_info: Information about revisited region
            
        Returns:
            Visualization image
        """
        vis_frame = frame.copy()
        
        # Overlay revisit mask if available
        if self.revisit_map.shape == (frame.shape[0], frame.shape[1]):
            overlay = np.zeros_like(frame)
            overlay[self.revisit_map] = [0, 0, 255]  # Red for revisited regions
            vis_frame = cv2.addWeighted(vis_frame, 1.0, overlay, 0.3, 0)
            
        # Add text information
        if revisit_info:
            cv2.putText(vis_frame, 
                      f"Revisit: Frame {revisit_info['keyframe_idx']} -> {revisit_info['current_idx']}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(vis_frame, 
                      f"Overlap: {revisit_info['overlap']:.2f}, Inliers: {revisit_info['inliers']}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return vis_frame
    
def predict(
    model,
    image,
    caption,
    box_threshold=0.35,
    text_threshold=0.25,
    device="cuda"
):
    """
    Patched version of GroundingDINO predict function that properly handles PIL Images.
    
    Args:
        model: GroundingDINO model
        image: PIL Image or torch tensor
        caption: text prompt
        box_threshold: box confidence threshold
        text_threshold: text confidence threshold
        device: device to run the model on
        
    Returns:
        boxes: (n, 4) tensor of bounding boxes in xyxy format, normalized to [0, 1]
        scores: (n) tensor of confidence scores
        phrases: list of matched text phrases
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to tensor
        image_tensor = transform_image(image)
        image_tensor = image_tensor.to(device)
    elif isinstance(image, torch.Tensor):
        image_tensor = image.to(device)
    else:
        raise ValueError("image must be a PIL Image or a torch tensor")
        
    # Run model
    with torch.no_grad():
        outputs = model(image_tensor, captions=[caption])
    
    # Extract predictions
    prediction_logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, num_classes)
    prediction_boxes = outputs["pred_boxes"][0]  # (num_queries, 4)
    
    # Filter predictions based on thresholds
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]
    
    # Get top-k predictions (already sorted by confidence)
    scores, phrase_indices = logits.max(dim=1)
    
    # Filter by text threshold
    mask = scores > text_threshold
    scores = scores[mask]
    boxes = boxes[mask]
    phrase_indices = phrase_indices[mask]
    
    # Convert token indices to phrases
    tokenizer = model.tokenizer
    phrases = [
        caption[caption.find(tokenizer.decode(p).strip()): caption.find(tokenizer.decode(p).strip()) + len(tokenizer.decode(p).strip())]
        if caption.find(tokenizer.decode(p).strip()) >= 0
        else ""
        for p in phrase_indices
    ]
    
    return boxes, scores, phrases

def transform_image(image_pil):
    """
    Transform PIL image to tensor format required by GroundingDINO.
    """
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil)
    return image_tensor.unsqueeze(0)

def annotate(image_pil, boxes, labels, colors=None, output_path=None):
    """
    Patched version of GroundingDINO annotate function that properly handles PIL Images.
    
    Args:
        image_pil: PIL Image
        boxes: (n, 4) tensor of bounding boxes in xyxy format, normalized to [0, 1]
        labels: list of text labels
        colors: list of colors for each label
        output_path: path to save annotated image
        
    Returns:
        annotated_image: PIL Image with annotations
    """
    if colors is None:
        colors = [[0, 255, 0]] * len(boxes)  # default: green
    
    # Convert to numpy for easier manipulation
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    
    # Create a copy of the image for annotation
    annotated_image = image_pil.copy()
    width, height = image_pil.size
    
    # Draw boxes and labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for i, (box, label, color) in enumerate(zip(boxes_np, labels, colors)):
        # Unnormalize box coordinates
        x0, y0, x1, y1 = box
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
        
        # Draw box
        draw.rectangle(((x0, y0), (x1, y1)), outline=tuple(color), width=2)
        
        # Draw label
        text_width, text_height = draw.textsize(label, font=font)
        draw.rectangle(((x0, y0), (x0 + text_width, y0 + text_height)), fill=tuple(color))
        draw.text((x0, y0), label, fill=(0, 0, 0), font=font)
    
    # Save if output path is provided
    if output_path:
        annotated_image.save(output_path)
        
    return annotated_image
# Import CoTracker for motion tracking
from cotracker.predictor import CoTrackerOnlinePredictor

# Import groundingDINO for improved object detection
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict



try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available. Install with: pip install huggingface_hub")

# Define process_frame_with_groundingdino function since it's not in the imports
def process_frame_with_groundingdino(model, image, text_prompt, box_threshold=0.35, text_threshold=0.25, revisit_mask=None, is_static_frame=False):
    """
    Process a frame with GroundingDINO to detect objects matching the text prompt.
    
    Args:
        model: The GroundingDINO model
        image: PIL Image or numpy array to process
        text_prompt: Text prompt describing objects to detect
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text matching
        revisit_mask: Optional mask of revisited regions to exclude from detection
        is_static_frame: Whether the frame is considered static
        
    Returns:
        boxes: Detected bounding boxes (xyxy format, normalized)
        scores: Confidence scores for each box
        phrases: Matched phrases for each box
    """
    try:
        # Skip processing if frame is static or has revisited regions
        if is_static_frame:
            logger.info("Skipping GroundingDINO processing for static frame")
            return torch.zeros((0, 4)), torch.zeros(0), []
            
        # Skip processing if frame has revisited regions
        if revisit_mask is not None and np.any(revisit_mask):
            logger.info("Skipping GroundingDINO processing for revisited frame")
            return torch.zeros((0, 4)), torch.zeros(0), []
            
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Process image with groundingdino
        boxes, scores, phrases = predict(
            model=model,
            image=image_pil,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        return boxes, scores, phrases
    except Exception as e:
        logger.error(f"Error in GroundingDINO processing: {str(e)}")
        import traceback
        logger.error(f"Detailed traceback: {traceback.format_exc()}")
        return None, None, None

cv2.ocl.setUseOpenCL(False)

def get_parser():
    # Get the directory of the script to use as base for finding the config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Parent directory of tool folder
    default_config = os.path.join(project_dir, 'config/cityscapes/cityscapes_pspnet18_sd.yaml')
    
    # Default paths for GroundingDINO based on the gradio_app.py file
    groundingdino_config_default = os.path.join(project_dir, 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    groundingdino_ckpt_default = None  # Will use huggingface download by default
    
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default=default_config,
                        help='config file')
    parser.add_argument('opts', help='see config/cityscapes/cityscapes_pspnet18_sd.yaml for all options', 
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--input_video', type=str, default=None,
                        help='Path to input video file for frame extraction')
    parser.add_argument('--video_output_path', type=str, default='segmentation_output.mp4',
                        help='Output video path for processed video')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Transparency value for segmentation overlay (0-1)')
    parser.add_argument('--rotate', type=int, default=0,
                        help='Rotation angle to be applied on each frame before processing (options: 0, 90, 180, 270)')
    parser.add_argument('--motion_threshold', type=float, default=0.3,
                        help='Threshold for motion detection (lower values: more sensitive)')
    parser.add_argument('--static_weight', type=float, default=0.0,
                        help='Weight applied to static scenes in label percentage calculation')
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Grid size for CoTracker point tracking')
    parser.add_argument('--spatial_tracking', action='store_true',
                        help='Enable spatial tracking to detect revisits')
    parser.add_argument('--revisit_threshold', type=int, default=100,
                        help='Minimum frame distance for revisit detection')
    parser.add_argument('--overlap_threshold', type=float, default=0.3,
                        help='Minimum overlap ratio to consider a frame as revisit')
    parser.add_argument('--revisit_discount', type=float, default=0.0,
                        help='Discount factor for labels in revisited regions (0-1, 0 = exclude completely)')
    parser.add_argument('--groundingdino_config', type=str, default=groundingdino_config_default,
                        help='Path to GroundingDINO model configuration file')
    parser.add_argument('--groundingdino_checkpoint', type=str, default=groundingdino_ckpt_default,
                        help='Path to GroundingDINO model checkpoint file')
    parser.add_argument('--groundingdino_hf_repo', type=str, default="ShilongLiu/GroundingDINO",
                        help="HuggingFace repo ID for GroundingDINO checkpoint")
    parser.add_argument('--groundingdino_hf_filename', type=str, default="groundingdino_swint_ogc.pth",
                        help="Filename of GroundingDINO checkpoint in HuggingFace repo")
    args = parser.parse_args()
    
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.input_video = args.input_video
    cfg.video_output_path = args.video_output_path
    cfg.alpha = args.alpha
    cfg.rotate = args.rotate
    cfg.motion_threshold = args.motion_threshold
    cfg.static_weight = args.static_weight
    cfg.grid_size = args.grid_size
    cfg.groundingdino_config = args.groundingdino_config
    cfg.groundingdino_checkpoint = args.groundingdino_checkpoint
    cfg.groundingdino_hf_repo = args.groundingdino_hf_repo
    cfg.groundingdino_hf_filename = args.groundingdino_hf_filename
    cfg.spatial_tracking = args.spatial_tracking
    cfg.revisit_threshold = args.revisit_threshold
    cfg.overlap_threshold = args.overlap_threshold
    cfg.revisit_discount = args.revisit_discount
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def predict_whole_img(net, image):
    with torch.no_grad():
        full_prediction_ = net(image.cuda())
    output = F.softmax(full_prediction_, dim=1)
    preds = output.cpu().data.numpy().transpose(0, 2, 3, 1)
    return preds

def get_class_names():
    return {
        0: 'road', 2: 'building', 8: 'vegetation'  # Only keep road, building, and vegetation (trees)
    }

def generate_video(frame_paths, output_path, frame_size, fps=30):
    logger.info(f"Generating video: {output_path}")
    start_time = perf_counter()
    
    # Read first frame to confirm dimensions and format
    # Use PIL to properly handle RGBA images
    first_image = Image.open(frame_paths[0])
    if first_image.mode == 'RGBA':
        logger.info("Processing transparent RGBA frames for video")
    
    actual_size = (first_image.width, first_image.height)
    logger.info(f"Output video dimensions: {actual_size}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, actual_size)
    
    for frame_path in frame_paths:
        # Read using PIL to properly handle RGBA
        img = Image.open(frame_path)
        
        if img.mode == 'RGBA':
            # Convert RGBA to RGB for video output
            # This will respect the alpha channel value we set
            rgb_img = Image.new('RGB', img.size, (0, 0, 0))  # Black background
            rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            
            # Convert to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        else:
            # Regular RGB image
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        
        if frame is not None:
            if frame.shape[:2][::-1] != actual_size:
                frame = cv2.resize(frame, actual_size)
            out.write(frame)
    
    out.release()
    logger.info(f"Video generation completed in {perf_counter() - start_time:.2f} seconds")

def get_video_rotation(filename):
    """Detect the rotation of the video from metadata using ffprobe."""
    try:
        import subprocess
        cmd = [
            'ffprobe',
            '-loglevel', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filename
        ]
        rotation = subprocess.check_output(cmd).decode('utf-8').strip()
        return int(rotation) if rotation else 0
    except:
        return 0

def initialize_cotracker(device="cuda"):
    """Initialize the CoTracker model for motion tracking."""
    try:
        # Load the model using torch.hub following official examples
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
        else:
            model = model.to("cpu")
        logger.info(f"CoTracker initialized successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize CoTracker: {str(e)}")
        logger.error(f"CoTracker error details: {type(e).__name__}")
        return None

def initialize_groundingdino(model_config_path, model_checkpoint_path=None, hf_repo_id=None, hf_filename=None, device="cuda"):
    """Initialize the GroundingDINO model for object detection and tracking.
    
    Args:
        model_config_path: Path to the model configuration file
        model_checkpoint_path: Path to local model checkpoint file
        hf_repo_id: HuggingFace repository ID if downloading from HF Hub
        hf_filename: Filename in the HuggingFace repository
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        The initialized GroundingDINO model
    """
    try:
        # Load configuration file
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        
        # Build the model
        logger.info(f"Building GroundingDINO model from config: {model_config_path}")
        model = build_model(args)
        
        # Load checkpoint - either from local path or HuggingFace Hub
        if model_checkpoint_path and os.path.isfile(model_checkpoint_path):
            logger.info(f"Loading GroundingDINO from local checkpoint: {model_checkpoint_path}")
            checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        elif HF_AVAILABLE and hf_repo_id and hf_filename:
            # Download from HuggingFace hub
            logger.info(f"Downloading GroundingDINO from HuggingFace: {hf_repo_id}/{hf_filename}")
            cache_file = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
            logger.info(f"Downloaded to cache: {cache_file}")
            
            checkpoint = torch.load(cache_file, map_location="cpu")
            log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            logger.info(f"Model loaded from HuggingFace: {log}")
        else:
            raise ValueError("Either a local checkpoint path or HuggingFace repo details must be provided")
        
        # Move model to specified device
        model = model.to(device)
        model.eval()
        
        logger.info(f"GroundingDINO initialized successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize GroundingDINO: {str(e)}")
        logger.error(f"GroundingDINO error details: {type(e).__name__}")
        return None

def calculate_motion_score(prev_points, curr_points, visibility):
    """Calculate motion score between two sets of tracked points."""
    if prev_points is None or curr_points is None:
        logger.debug("calculate_motion_score: No previous or current points available")
        return 1.0  # Assume maximum motion if we don't have previous points
    
    # Filter points by visibility
    visible_mask = visibility[0, -1] > 0.5
    num_visible = torch.sum(visible_mask).item()
    
    logger.debug(f"calculate_motion_score: Total points: {visible_mask.shape[0]}, Visible points: {num_visible}")
    
    if not torch.any(visible_mask):
        logger.debug("calculate_motion_score: No visible points")
        return 1.0  # No visible points
    
    # Calculate displacement of visible points
    prev_visible = prev_points[0, -1, visible_mask]
    curr_visible = curr_points[0, -1, visible_mask]
    
    if len(prev_visible) == 0:
        logger.debug("calculate_motion_score: Empty visible points array")
        return 1.0
    
    # Calculate euclidean distance for each point
    distances = torch.norm(curr_visible - prev_visible, dim=1)
    
    # Log histogram of distances to understand distribution
    if len(distances) > 0:
        hist_counts = torch.histc(distances, bins=10, min=0, max=50).cpu().numpy().astype(int)
        logger.debug(f"Distance histogram: {hist_counts.tolist()}")
    
    # Use mean distance as motion score
    mean_distance = torch.mean(distances).item()
    max_distance = torch.max(distances).item() if len(distances) > 0 else 0
    
    # More sophisticated normalization: consider both mean and max displacement
    # and normalize based on a fraction of image dimension
    frame_dimension = 50.0  # Normalization factor (can be adjusted)
    normalized_score = min(mean_distance / frame_dimension, 1.0)
    
    logger.debug(f"Motion calculation: mean_dist={mean_distance:.2f}, max_dist={max_distance:.2f}, score={normalized_score:.4f}")
    
    return normalized_score

def detect_scene_change(prev_frame, curr_frame, threshold=30):
    """Detect if there's a significant scene change between frames."""
    if prev_frame is None or curr_frame is None:
        return False
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    
    # Compare histograms
    similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    
    # If similarity is low, it's a scene change
    return similarity < (1 - threshold/100)

def _process_cotracker_step(cotracker, current_window, is_first_step, grid_size, grid_query_frame, device):
    """Helper function to process a step in CoTracker."""
    try:
        video_chunk = torch.tensor(
            np.stack(current_window), 
            device=device
        ).float().permute(0, 3, 1, 2)[None]
        
        pred_tracks, pred_visibility = cotracker(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame
        )
        return pred_tracks, pred_visibility
    except Exception as e:
        logger.error(f"Error in CoTracker step: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

class InstanceTracker:
    """
    Tracks object instances across video frames using spatial and appearance similarity.
    
    This class assigns and maintains unique IDs for objects detected in consecutive frames,
    ensuring consistent tracking of the same object instance across the video.
    """
    
    def __init__(self, iou_threshold=0.3, max_disappeared=30, min_confidence=0.25):
        """
        Initialize the instance tracker.
        
        Args:
            iou_threshold: Threshold for considering boxes as matching between frames
            max_disappeared: Maximum number of frames an object can disappear before considered lost
            min_confidence: Minimum confidence score to consider an object for tracking
        """
        self.next_object_id = 1  # Start object IDs at 1
        self.objects = {}  # Tracked objects dict: {object_id: {'box': box, 'class_id': class_id, 'score': score, 'last_seen': frame_idx}}
        self.disappeared = {}  # Counter for disappeared objects
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.min_confidence = min_confidence
        
        # Stats for each class
        self.class_instance_counts = defaultdict(int)
        self.unique_instances_seen = set()  # Set of all unique object IDs seen
    
    def update(self, frame_idx, detections_by_class):
        """
        Update object tracking with new detections.
        
        Args:
            frame_idx: Current frame index
            detections_by_class: Dictionary of {class_id: {'boxes': boxes, 'scores': scores}}
                                 where boxes is numpy array of [x0, y0, x1, y1] normalized coordinates
        
        Returns:
            Dictionary mapping object IDs to {'box': box, 'class_id': class_id, 'score': score}
        """
        # If no objects are being tracked yet, initialize with first detections
        if len(self.objects) == 0:
            return self._register_initial_objects(frame_idx, detections_by_class)
        
        # Prepare all current detections for matching
        current_boxes = []
        current_class_ids = []
        current_scores = []
        
        for class_id, detection_data in detections_by_class.items():
            boxes = detection_data['boxes']
            scores = detection_data['scores']
            
            for box_idx, (box, score) in enumerate(zip(boxes, scores)):
                if score >= self.min_confidence:  # Only track objects above confidence threshold
                    current_boxes.append(box)
                    current_class_ids.append(class_id)
                    current_scores.append(score)
        
        # If no detections in this frame, mark all existing objects as disappeared
        if len(current_boxes) == 0:
            return self._handle_disappeared_objects(frame_idx)
        
        # If we have existing objects, match new detections to existing objects
        return self._match_and_update_objects(frame_idx, current_boxes, current_class_ids, current_scores)
    
    def _register_initial_objects(self, frame_idx, detections_by_class):
        """Register objects in the first frame or when no objects are currently tracked."""
        result = {}
        
        for class_id, detection_data in detections_by_class.items():
            boxes = detection_data['boxes']
            scores = detection_data['scores']
            
            for box_idx, (box, score) in enumerate(zip(boxes, scores)):
                if score >= self.min_confidence:
                    object_id = self.next_object_id
                    self.next_object_id += 1
                    
                    # Register the new object
                    self.objects[object_id] = {
                        'box': box.copy(),
                        'class_id': class_id,
                        'score': score,
                        'last_seen': frame_idx
                    }
                    
                    self.disappeared[object_id] = 0
                    
                    # Update class statistics
                    self.class_instance_counts[class_id] += 1
                    self.unique_instances_seen.add(object_id)
                    
                    result[object_id] = {
                        'box': box.copy(),
                        'class_id': class_id,
                        'score': score
                    }
        
        return result
    
    def _handle_disappeared_objects(self, frame_idx):
        """Handle the case when no detections are made in the current frame."""
        result = {}
        
        # Increment disappeared counter for all objects
        for object_id in list(self.objects.keys()):
            self.disappeared[object_id] += 1
            
            # If object has been missing too long, remove it
            if self.disappeared[object_id] > self.max_disappeared:
                del self.objects[object_id]
                del self.disappeared[object_id]
            else:
                # Keep the object but mark it as not detected in this frame
                obj = self.objects[object_id]
                result[object_id] = {
                    'box': obj['box'],
                    'class_id': obj['class_id'],
                    'score': obj['score'],
                    'disappeared': True
                }
        
        return result
    
    def _match_and_update_objects(self, frame_idx, current_boxes, current_class_ids, current_scores):
        """Match current detections to existing tracked objects."""
        # Calculate IoU between existing and new objects
        if not current_boxes or len(current_boxes) == 0:
            return self._handle_disappeared_objects(frame_idx)
            
        current_boxes = np.array(current_boxes)
        current_class_ids = np.array(current_class_ids)
        current_scores = np.array(current_scores)
        
        existing_object_ids = list(self.objects.keys())
        existing_boxes = np.array([self.objects[obj_id]['box'] for obj_id in existing_object_ids])
        existing_class_ids = np.array([self.objects[obj_id]['class_id'] for obj_id in existing_object_ids])
        
        # Initialize matrix to track whether each detection has been matched
        used_detections = [False] * len(current_boxes)
        
        result = {}
        
        # Calculate IoU between all existing and current boxes
        # For each existing object:
        for i, object_id in enumerate(existing_object_ids):
            existing_box = existing_boxes[i]
            existing_class = existing_class_ids[i]
            
            # Only try to match to objects of the same class
            best_iou = 0
            best_detection_idx = -1
            
            for j, (current_box, current_class) in enumerate(zip(current_boxes, current_class_ids)):
                # Skip if already matched or different class
                if used_detections[j] or current_class != existing_class:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(existing_box, current_box)
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = j
            
            # If we found a match
            if best_detection_idx >= 0:
                # Update object with new position
                matched_box = current_boxes[best_detection_idx].copy()
                matched_score = current_scores[best_detection_idx]
                
                self.objects[object_id] = {
                    'box': matched_box,
                    'class_id': existing_class,  # Class remains the same
                    'score': matched_score,
                    'last_seen': frame_idx
                }
                
                # Reset disappeared counter
                self.disappeared[object_id] = 0
                
                # Mark this detection as used
                used_detections[best_detection_idx] = True
                
                # Add to result
                result[object_id] = {
                    'box': matched_box,
                    'class_id': existing_class,
                    'score': matched_score
                }
            else:
                # No match found, increment disappeared counter
                self.disappeared[object_id] += 1
                
                # If object has been missing too long, remove it
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
                else:
                    # Keep object but mark as disappeared
                    obj = self.objects[object_id]
                    result[object_id] = {
                        'box': obj['box'],
                        'class_id': obj['class_id'],
                        'score': obj['score'],
                        'disappeared': True
                    }
        
        # Register new objects for detections that weren't matched
        for i, used in enumerate(used_detections):
            if not used:
                object_id = self.next_object_id
                self.next_object_id += 1
                
                # Register new object
                box = current_boxes[i].copy()
                class_id = current_class_ids[i]
                score = current_scores[i]
                
                self.objects[object_id] = {
                    'box': box,
                    'class_id': class_id,
                    'score': score,
                    'last_seen': frame_idx
                }
                
                self.disappeared[object_id] = 0
                
                # Update class statistics
                self.class_instance_counts[class_id] += 1
                self.unique_instances_seen.add(object_id)
                
                result[object_id] = {
                    'box': box,
                    'class_id': class_id,
                    'score': score
                }
        
        return result
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        
        Args:
            box1, box2: Boxes in format [x0, y0, x1, y1] (normalized coordinates)
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection coordinates
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou
    
    def get_statistics(self):
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        total_instances = len(self.unique_instances_seen)
        
        # Calculate percentages
        percentages = {}
        for class_id, count in self.class_instance_counts.items():
            if total_instances > 0:
                percentages[class_id] = (count / total_instances) * 100
            else:
                percentages[class_id] = 0
                
        return {
            'total_unique_instances': total_instances,
            'class_instance_counts': dict(self.class_instance_counts),
            'class_percentages': percentages
        }

def process_video_with_motion_tracking(input_video, model, transform_fn, colors, output_folder, 
                                       video_output_path, fps=30, alpha=1, motion_threshold=0.3,
                                       static_weight=0.2, grid_size=10, groundingdino_model=None,
                                       spatial_tracking=False, revisit_threshold=100, 
                                       overlap_threshold=0.3, revisit_discount=0.0):
    """Process video with motion tracking to handle static scenes."""
    global logger
    start_time = perf_counter()
    logger.info(f"Starting video processing with motion tracking at {time.strftime('%H:%M:%S')}")
    
    # Detect video rotation from metadata
    video_rotation = get_video_rotation(input_video)
    logger.info(f"Detected video rotation: {video_rotation} degrees")
    
    # Initialize CoTracker for motion tracking
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = initialize_cotracker(device)
    if cotracker is None:
        logger.warning("CoTracker initialization failed. Falling back to standard processing.")
    
    # Initialize spatial tracker if enabled
    spatial_tracker = None
    if spatial_tracking:
        logger.info("Initializing spatial tracker for revisit detection")
        spatial_tracker = SpatialTracker(
            revisit_threshold=revisit_threshold,
            overlap_threshold=overlap_threshold,
            debug=True  # Set to True for more detailed logging
        )
    
    # Initialize instance tracker for unique object tracking
    instance_tracker = InstanceTracker(
        iou_threshold=0.3,
        max_disappeared=30,
        min_confidence=0.25
    )
    
    # Prepare text prompts for GroundingDINO
    dino_text_prompts = {
        0: "road . street . path . highway",
        2: "building . architecture . structure . house . skyscraper",
        8: "vegetation . trees . plants . grass . forest"
    }
    
    # DINO box threshold and text threshold
    box_threshold = 0.25
    text_threshold = 0.25
    
    # Class name mapping for display and statistics
    class_names = get_class_names()
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_video}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
    
    # Calculate a more appropriate look-ahead size based on FPS
    # For a 30fps video, this would be ~15 frames (half a second)
    look_ahead_size = max(int(fps * 0.5), 15)  # At least 15 frames or half a second
    logger.info(f"Using look-ahead buffer of {look_ahead_size} frames ({look_ahead_size/fps:.2f} seconds)")
    
    ret, frame = cap.read()
    if not ret:
        logger.error("Could not read the first frame from the video.")
        return None
    
    # Store original dimensions
    original_height, original_width = frame.shape[:2]
    
    # Calculate dimensions after rotation
    if video_rotation in [90, 270]:
        frame_width, frame_height = original_height, original_width
    else:
        frame_width, frame_height = original_width, original_height
    
    frame_size = (frame_width, frame_height)
    logger.info(f"Frame size after rotation: {frame_size}")
    
    # Apply additional rotation if specified by --rotate argument
    total_rotation = (video_rotation + args.rotate) % 360
    
    # Reset video capture to start
    cap.release()
    cap = cv2.VideoCapture(input_video)
    
    # Tracking variables
    frames_paths = []
    frame_idx = 0
    frame_processing_times = []
    prev_frame = None
    prev_tracks = None
    prev_visibility = None
    
    # Store label statistics with motion weighting
    label_pixels = defaultdict(float)  # Overall label pixels (motion frames only)
    static_label_pixels = defaultdict(float)  # Labels specifically in static frames
    static_frame_count = 0  # Count of static frames
    total_weighted_pixels = 0
    
    # Percentage to redistribute from dominant static labels
    redistribution_percentage = 0.075  # Redistribute 7.5% from dominant static label
    
    # Add label confidence tracking mechanism 
    confidence_threshold = 0.75  # Confidence threshold for stable label detection
    confidence_frames = 25  # Number of frames required for a label to be considered stable
    label_confidence_maps = {}  # Track confidence for each pixel position and label
    stable_label_map = None  # Map of stable labels that have persisted for confidence_frames
    
    # Buffer to store frames for CoTracker
    frame_buffer = []
    
    # Look-ahead buffer for frames and their metadata
    buffered_frames = []  # Store frame data and metadata
    future_motion_scores = []  # Store motion scores for upcoming frames
    
    # Initialize motion state
    is_static_scene = False
    static_frames_count = 0
    curr_static_sequence = 0
    static_sequence_id = 0
    processed_static_sequences = set()
    motion_scores = []
    
    # CoTracker window size and step size
    window_size = cotracker.step * 2 if cotracker else 0
    
    # Frames and predictions storage
    frame_predictions = []
    frame_weights = []
    frames_data = []
    
    # Store GroundingDINO detections 
    dino_detections = {}
    
    # Spatial tracking variables
    revisit_info = None
    revisit_mask = None
    revisit_count = 0
    revisited_regions = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # How often to run GroundingDINO (every N frames)
    dino_frequency = min(30, fps)  # Run DINO once per second or less often
    logger.info(f"GroundingDINO will run every {dino_frequency} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = perf_counter()

        # Apply rotation if needed
        if total_rotation != 0:
            if total_rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif total_rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif total_rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Ensure frame has correct dimensions
        if frame.shape[:2] != (frame_height, frame_width):
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Append to frame buffer for motion tracking
        frame_buffer.append(frame)
        
        # Process frames in batches for CoTracker
        motion_score = 1.0  # Default: full motion
        
        # Only process motion tracking when buffer has enough frames
        if cotracker and len(frame_buffer) >= window_size and frame_idx % cotracker.step == 0:
            try:
                # Extract window of frames for tracking
                current_window = frame_buffer[-window_size:]
                
                # Following the official example pattern
                logger.info(f"Frame {frame_idx}: Processing chunk with {len(current_window)} frames")
                
                is_first_step = (frame_idx <= cotracker.step*2)
                logger.info(f"Frame {frame_idx}: Processing with is_first_step={is_first_step}")
                
                # Process with the helper function that matches the cotracker_example.py pattern
                pred_tracks, pred_visibility = _process_cotracker_step(
                    cotracker, 
                    current_window, 
                    is_first_step, 
                    grid_size, 
                    grid_query_frame=0, 
                    device=device
                )
                
                # Check if model returned valid outputs
                if pred_tracks is None or pred_visibility is None:
                    logger.error(f"Frame {frame_idx}: CoTracker returned None values")
                    motion_score = 1.0
                    is_static_scene = False  # Assume not static if tracking failed
                else:
                    logger.info(f"Frame {frame_idx}: Tracks shape: {pred_tracks.shape}, Visibility shape: {pred_visibility.shape}")
                    
                    # Calculate motion score based on tracked points
                    if not is_first_step and prev_tracks is not None:
                        motion_score = calculate_motion_score(prev_tracks, pred_tracks, pred_visibility)
                        motion_scores.append(motion_score)
                        future_motion_scores.append(motion_score)  # Also store for future prediction
                        logger.info(f"Frame {frame_idx}: Motion score = {motion_score:.4f} (threshold = {motion_threshold})")
                    
                    # Update previous tracks for next iteration
                    prev_tracks = pred_tracks
                    prev_visibility = pred_visibility
                    
                    # Detect if scene is static
                    was_static = is_static_scene
                    is_static_scene = motion_score < motion_threshold
                    
                    if is_static_scene:
                        # This is a static frame
                        if not was_static:  # First static frame in this sequence
                            static_sequence_id += 1
                            curr_static_sequence = static_sequence_id
                            logger.info(f"Static scene #{static_sequence_id} detected at frame {frame_idx}, motion score: {motion_score:.4f}")
                        static_frames_count += 1
                    else:
                        # This is a motion frame
                        if was_static:
                            logger.info(f"Motion detected at frame {frame_idx} after {static_frames_count} static frames")
                            static_frames_count = 0
                            curr_static_sequence = 0  # Reset current static sequence ID
                
            except Exception as e:
                logger.error(f"Error in motion tracking at frame {frame_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                motion_score = 1.0  # Assume full motion on error
                is_static_scene = False  # Assume not static if error occurs
        
        # Scene change detection as a fallback/supplement
        if prev_frame is not None and detect_scene_change(prev_frame, frame):
            logger.info(f"Scene change detected at frame {frame_idx}")
            motion_score = 1.0  # Full weight for scene changes
            is_static_scene = False
            static_frames_count = 0
            curr_static_sequence = 0  # Reset current static sequence ID
        
        # Store current frame for next iteration
        prev_frame = frame.copy()
        
        # Prepare frame data
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frame_np = np.array(pil_img)
        dummy_label = np.zeros(frame_np.shape[:2], dtype=np.uint8)
        input_tensor, _ = transform_fn(frame_np, dummy_label)
        input_tensor = input_tensor.unsqueeze(0).cuda()

        # Predict segmentation using PSPNet
        prediction = predict_whole_img(model, input_tensor)
        prediction = np.argmax(prediction, axis=3)[0]
        
        # Run GroundingDINO on selected frames
        if groundingdino_model and frame_idx % dino_frequency == 0:
            logger.info(f"Frame {frame_idx}: Running GroundingDINO object detection")
            
            # Process each class with its corresponding text prompt
            dino_frame_results = {}
            
            for label_id, prompt in dino_text_prompts.items():
                # Process the current frame with GroundingDINO using class-specific prompt
                boxes, scores, phrases = process_frame_with_groundingdino(
                    groundingdino_model,
                    pil_img,
                    prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    revisit_mask=revisit_mask,  # Pass revisit mask to exclude revisited regions
                    is_static_frame=is_static_scene  # Skip processing for static frames
                )
                
                if boxes is not None and len(boxes) > 0:
                    # Store results for this class
                    dino_frame_results[label_id] = {
                        'boxes': boxes.cpu().numpy(),
                        'scores': scores.cpu().numpy(),
                        'phrases': phrases
                    }
                    
                    # Log detection results
                    logger.info(f"  Class {class_names[label_id]}: {len(boxes)} detections")
                    
                    # Create a mask from bounding boxes for this class
                    h, w = frame_np.shape[:2]
                    class_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Fill in mask from denormalized bounding boxes
                    for box in boxes:
                        # Convert normalized coordinates to pixels
                        x0, y0, x1, y1 = box.cpu().numpy()
                        x0, x1 = int(x0 * w), int(x1 * w)
                        y0, y1 = int(y0 * h), int(y1 * h)
                        
                        # Clamp values to image boundaries
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(w-1, x1), min(h-1, y1)
                        
                        # Fill the bounding box area in the mask
                        class_mask[y0:y1, x0:x1] = 1
                    
                    # Enhance segmentation by merging GroundingDINO detections
                    # Only modify/enhance areas that GroundingDINO detected with high confidence
                    confident_dino_areas = (class_mask == 1)
                    
                    # Update the prediction for this class where DINO detected it
                    # but keep the original for other segments
                    if confident_dino_areas.any():
                        # Create a mask that represents where we'll apply the GroundingDINO enhancement
                        # We'll only replace low-confidence segmentations with high-confidence DINO detections
                        enhancement_mask = confident_dino_areas
                        
                        # Apply the enhancement to the prediction
                        prediction[enhancement_mask] = label_id
                        logger.info(f"  Enhanced prediction for {class_names[label_id]} using {np.sum(enhancement_mask)} pixels from GroundingDINO")
            
            # Store the complete DINO results for this frame
            if dino_frame_results:
                dino_detections[frame_idx] = dino_frame_results
                
                # Update instance tracker with new detections
                tracked_objects = instance_tracker.update(frame_idx, dino_frame_results)
                
                # Log tracking statistics periodically
                if frame_idx % (dino_frequency * 5) == 0:
                    stats = instance_tracker.get_statistics()
                    logger.info(f"Instance Tracking Stats (frame {frame_idx}):")
                    logger.info(f"  Total unique instances: {stats['total_unique_instances']}")
                    
                    # Log stats for each class
                    for class_id, count in stats['class_instance_counts'].items():
                        class_name = class_names.get(class_id, f"Class {class_id}")
                        percentage = stats['class_percentages'].get(class_id, 0)
                        logger.info(f"  {class_name}: {count} instances ({percentage:.1f}%)")
        
        # Filter prediction to include only target classes
        valid_labels = list(class_names.keys())
        mask_valid = np.isin(prediction, valid_labels)
        
        # Create filtered prediction
        filtered_prediction = np.copy(prediction)
        filtered_prediction[~mask_valid] = 255  # Set non-valid labels to background
        
        # Update the spatial tracker for revisit detection if enabled
        if spatial_tracking and spatial_tracker is not None:
            # Use the filtered prediction mask as our label mask
            label_mask = np.ones_like(filtered_prediction, dtype=np.uint8)
            label_mask[filtered_prediction == 255] = 0  # Exclude background from tracking
            
            # Update spatial tracker with current frame
            revisit_info, revisit_mask = spatial_tracker.update(frame, frame_idx, label_mask)
            
            if revisit_info is not None:
                revisit_count += 1
                logger.info(f"Frame {frame_idx}: Detected revisit to frame {revisit_info['keyframe_idx']} "
                          f"(overlap: {revisit_info['overlap']:.2f}, inliers: {revisit_info['inliers']})")
                
                # Update cumulative revisited regions
                if revisit_mask is not None and revisit_mask.shape == (frame_height, frame_width):
                    revisited_regions = np.logical_or(revisited_regions, revisit_mask).astype(np.uint8)
        else:
            revisit_info = None
            revisit_mask = None
        
        # Store the frame data in the buffer with all necessary information
        buffered_frames.append({
            'frame_idx': frame_idx,
            'frame': frame_rgb,
            'prediction': filtered_prediction,
            'is_static': is_static_scene,
            'motion_score': motion_score,
            'has_dino': frame_idx in dino_detections,  # Flag if this frame has DINO detections
            'tracked_objects': tracked_objects if frame_idx in dino_detections else {},
            'revisit_info': revisit_info,  # Add revisit information
            'revisit_mask': revisit_mask  # Add revisit mask
        })
        
        # Process buffered frames if we have enough look-ahead information or reached end of video
        if len(buffered_frames) > look_ahead_size or not ret:
            # Get the oldest frame from buffer
            frame_data = buffered_frames[0]
            buffered_idx = frame_data['frame_idx']
            
            # Enhanced look-ahead logic:
            # 1. Check if any future frame is marked static
            future_static = any(f['is_static'] for f in buffered_frames[1:])
            
            # 2. Check if future motion scores are trending downward (approaching static)
            # This helps identify frames that are transitioning to static but haven't crossed threshold yet
            future_scores = [f['motion_score'] for f in buffered_frames[1:] if 'motion_score' in f]
            
            # If we have at least 3 future scores, detect if they're trending toward static
            trending_to_static = False
            if len(future_scores) >= 3:
                # Check if motion scores are consistently decreasing toward static threshold
                decreasing_trend = all(a >= b for a, b in zip(future_scores[:-1], future_scores[1:]))
                approaching_static = any(score < motion_threshold * 1.5 for score in future_scores)
                trending_to_static = decreasing_trend and approaching_static
                
                if trending_to_static:
                    logger.info(f"Frame {buffered_idx}: Detected trending toward static (scores: {future_scores})")
            
            # Determine if this frame should be counted based on future frames
            if future_static or frame_data['is_static'] or trending_to_static:
                # Either current frame or any future frame is static or trending to static
                count_pixels = False
                frame_weight = 0.0
                status = "Static (predicted)" if trending_to_static else "Static (future)" if future_static else "Static"
            else:
                # Current frame and all future frames are not static
                count_pixels = True
                frame_weight = 1.0
                status = "Motion"
                
            logger.info(f"Processing buffered frame {buffered_idx}: {status}, weight={frame_weight}")
            
            # Store frame weight
            frame_weights.append(frame_weight)
            frame_predictions.append(frame_data['prediction'])
            
            # Get revisit mask information from spatial tracker
            curr_revisit_mask = frame_data.get('revisit_mask')
            curr_revisit_info = frame_data.get('revisit_info')
            
            # Update label statistics with weighted pixel counts and handling revisits
            if count_pixels:
                for label_id in class_names.keys():
                    # Create mask for this label
                    label_mask = (frame_data['prediction'] == label_id)
                    
                    # Apply revisit discount to pixels that are in revisited regions
                    if spatial_tracking and curr_revisit_mask is not None and curr_revisit_mask.any():
                        # Calculate effective pixel count with revisit discount
                        revisited_label_pixels = np.logical_and(label_mask, curr_revisit_mask).sum()
                        non_revisited_label_pixels = np.logical_and(label_mask, ~curr_revisit_mask).sum()
                        
                        # Apply discount factor to revisited regions (0 = exclude completely)
                        effective_pixel_count = non_revisited_label_pixels + revisited_label_pixels * revisit_discount
                    else:
                        effective_pixel_count = np.sum(label_mask)
                        
                    # Apply motion weight
                    weighted_count = effective_pixel_count * frame_weight
                    label_pixels[label_id] += weighted_count
                    total_weighted_pixels += weighted_count if label_id in valid_labels else 0
            else:
                # Update static label statistics
                for label_id in class_names.keys():
                    # Create mask for this label
                    label_mask = (frame_data['prediction'] == label_id)
                    
                    # Apply revisit discount to pixels that are in revisited regions for static frames too
                    if spatial_tracking and curr_revisit_mask is not None and curr_revisit_mask.any():
                        # Calculate effective pixel count with revisit discount
                        revisited_label_pixels = np.logical_and(label_mask, curr_revisit_mask).sum()
                        non_revisited_label_pixels = np.logical_and(label_mask, ~curr_revisit_mask).sum()
                        
                        # Apply discount factor to revisited regions
                        pixel_count = non_revisited_label_pixels + revisited_label_pixels * revisit_discount
                    else:
                        pixel_count = np.sum(label_mask)
                    
                    static_label_pixels[label_id] += pixel_count
                static_frame_count += 1

            # Colorize the segmentation mask
            _, colored_mask = colorize(np.uint8(frame_data['prediction']), colors)
            colored_mask = cv2.resize(colored_mask, (frame_width, frame_height))
            
            # Get the original frame for blending
            frame_rgb = cv2.resize(frame_data['frame'], (frame_width, frame_height))
            
            # Apply alpha blending between original frame and colored mask
            clean_output = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)
            
            # Save the blended image
            frame_filename = os.path.join(output_folder, f"frame_{buffered_idx:06d}.png")
            Image.fromarray(clean_output.astype('uint8')).save(frame_filename)
            frames_paths.append(frame_filename)
            
            # Track processing time
            frame_time = perf_counter() - frame_start
            frame_processing_times.append(frame_time)
            
            # Remove the processed frame from buffer
            buffered_frames.pop(0)
            
            # Keep the future_motion_scores buffer limited to our look-ahead window
            if len(future_motion_scores) > look_ahead_size * 2:  # Keep twice the look-ahead buffer for trend analysis
                future_motion_scores.pop(0)
                
            if buffered_idx % 10 == 0:
                logger.info(f"Processed {buffered_idx} frames, motion score: {motion_score:.4f}")
        
        frame_idx += 1

    cap.release()
    
    # Process any remaining frames in the buffer
    for frame_data in buffered_frames:
        buffered_idx = frame_data['frame_idx']
        
        # At the end of video, mark as motion if the frame itself is not static
        count_pixels = not frame_data['is_static']
        frame_weight = 1.0 if count_pixels else 0.0
        status = "Motion" if count_pixels else "Static"
            
        logger.info(f"Processing remaining buffered frame {buffered_idx}: {status}, weight={frame_weight}")
        
        # Store frame weight
        frame_weights.append(frame_weight)
        frame_predictions.append(frame_data['prediction'])
        
        # Get revisit mask information
        curr_revisit_mask = frame_data.get('revisit_mask')
        curr_revisit_info = frame_data.get('revisit_info')
        
        # Update label statistics with weighted pixel counts
        if count_pixels:
            for label_id in class_names.keys():
                # Create mask for this label
                label_mask = (frame_data['prediction'] == label_id)
                
                # Apply revisit discount to pixels that are in revisited regions
                if spatial_tracking and curr_revisit_mask is not None and curr_revisit_mask.any():
                    # Calculate effective pixel count with revisit discount
                    revisited_label_pixels = np.logical_and(label_mask, curr_revisit_mask).sum()
                    non_revisited_label_pixels = np.logical_and(label_mask, ~curr_revisit_mask).sum()
                    
                    # Apply discount factor to revisited regions
                    effective_pixel_count = non_revisited_label_pixels + revisited_label_pixels * revisit_discount
                else:
                    effective_pixel_count = np.sum(label_mask)
                    
                # Apply motion weight
                weighted_count = effective_pixel_count * frame_weight
                label_pixels[label_id] += weighted_count
                total_weighted_pixels += weighted_count if label_id in valid_labels else 0
        else:
            # Update static label statistics
            for label_id in class_names.keys():
                # Create mask for this label
                label_mask = (frame_data['prediction'] == label_id)
                
                # Apply revisit discount to pixels that are in revisited regions
                if spatial_tracking and curr_revisit_mask is not None and curr_revisit_mask.any():
                    # Calculate effective pixel count with revisit discount
                    revisited_label_pixels = np.logical_and(label_mask, curr_revisit_mask).sum()
                    non_revisited_label_pixels = np.logical_and(label_mask, ~curr_revisit_mask).sum()
                    
                    # Apply discount factor to revisited regions
                    pixel_count = non_revisited_label_pixels + revisited_label_pixels * revisit_discount
                else:
                    pixel_count = np.sum(label_mask)
                
                static_label_pixels[label_id] += pixel_count
            static_frame_count += 1

        # Colorize and blend the segmentation mask with original frame
        _, colored_mask = colorize(np.uint8(frame_data['prediction']), colors)
        colored_mask = cv2.resize(colored_mask, (frame_width, frame_height))
        
        # Get the original frame for blending
        frame_rgb = cv2.resize(frame_data['frame'], (frame_width, frame_height))
        
        # Apply alpha blending between original frame and colored mask
        clean_output = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)
        
        # Save the blended image
        frame_filename = os.path.join(output_folder, f"frame_{buffered_idx:06d}.png")
        Image.fromarray(clean_output.astype('uint8')).save(frame_filename)
        frames_paths.append(frame_filename)
    
    # Generate the output video if requested
    if len(frames_paths) > 0 and video_output_path:
        generate_video(frames_paths, video_output_path, frame_size, fps)
    
    # Get final instance tracking statistics
    tracking_stats = instance_tracker.get_statistics()
    
    # ADDED: Save tracking statistics to CSV file
    csv_output_path = os.path.splitext(video_output_path)[0] + "_labels.csv"
    with open(csv_output_path, 'w') as csv_file:
        # Write header
        csv_file.write("Label,Name,Percentage\n")
        
        # Sort classes by percentage for output
        sorted_classes = sorted(
            tracking_stats['class_percentages'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Write data
        for class_id, percentage in sorted_classes:
            class_name = class_names.get(class_id, f"Unknown")
            csv_file.write(f"{class_id},{class_name},{percentage:.2f}\n")
    
    # ADDED: Save tracking statistics to JSON file
    import json
    json_output_path = os.path.splitext(video_output_path)[0] + "_labels.json"
    
    json_data = []
    for class_id, percentage in sorted_classes:
        # Convert NumPy types to native Python types to ensure JSON serialization works
        class_id_py = int(class_id) if hasattr(class_id, 'item') else class_id
        percentage_py = float(percentage) if hasattr(percentage, 'item') else percentage
        
        json_data.append({
            "Label": class_id_py,
            "Name": class_names.get(class_id, f"Unknown"),
            "Percentage": round(percentage_py, 2)
        })
    
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    
    # Display final statistics
    logger.info("\n===== STATISTICS =====")
    # logger.info(f"Total Unique Instances Detected: {tracking_stats['total_unique_instances']}")
    logger.info(f"Statistics saved to {csv_output_path} and {json_output_path}")
    
    # Sort classes by count for display
    sorted_classes = sorted(
        tracking_stats['class_instance_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Display statistics for each class
    for class_id, count in sorted_classes:
        class_name = class_names.get(class_id, f"Class {class_id}")
        percentage = tracking_stats['class_percentages'].get(class_id, 0)
        logger.info(f"{class_name}: {percentage:.1f}%")
    
    logger.info("====================================\n")
    
    # # Log spatial tracking statistics if enabled
    # if spatial_tracking and spatial_tracker is not None:
    #     logger.info("\n===== SPATIAL TRACKING STATISTICS =====")
    #     logger.info(f"Total revisits detected: {revisit_count}")
    #     logger.info(f"Total keyframes stored: {len(spatial_tracker.keyframes)}")
        
    #     # Calculate what percentage of the frame area was revisited at least once
    #     revisited_area_percentage = np.mean(revisited_regions) * 100
    #     logger.info(f"Total area revisited: {revisited_area_percentage:.2f}%")
    #     logger.info(f"Discount factor for revisited regions: {revisit_discount:.2f}")
        
    #     if revisit_discount == 0:
    #         logger.info("Revisited regions were completely excluded from label counting")
    #     elif revisit_discount < 1.0:
    #         logger.info(f"Revisited regions were counted with {revisit_discount:.2f} weight")
    #     logger.info("====================================\n")
    
    # Calculate and return total processing time
    total_processing_time = perf_counter() - start_time
    logger.info(f"Total video processing time: {total_processing_time:.2f} seconds")
    logger.info(f"Average time per frame: {np.mean(frame_processing_times):.2f} seconds")
    
    return total_processing_time

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    
    pipeline_start = perf_counter()
    logger.info(f"Starting pipeline at {time.strftime('%H:%M:%S')}")
    
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gen_gpu)
    
    # Get project directory for resolving paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Helper function to resolve paths
    def resolve_path(path):
        if not path:
            return path
        if not os.path.isabs(path):
            resolved_path = os.path.join(project_dir, path)
            logger.info(f"Resolving path: '{path}' to '{resolved_path}'")
            return resolved_path
        return path
    
    # Resolve paths
    colors_path = resolve_path(args.colors_path)
    ckpt_path = resolve_path(args.ckpt_path)
    save_folder = resolve_path(args.save_folder)
    
    # Data preprocessing setup
    value_scale = 255
    mean = [item * value_scale for item in [0.485, 0.456, 0.406]]
    std = [item * value_scale for item in [0.229, 0.224, 0.225]]
    colors = np.loadtxt(colors_path).astype('uint8')

    # Model setup
    from model.pspnet_18 import PSPNet
    model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, 
                   flow=False, pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    cudnn.benchmark = True

    # Load checkpoint
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(transfer_ckpt(checkpoint), strict=False)
    else:
        raise RuntimeError(f"=> no checkpoint found at '{ckpt_path}'")
    
    # Initialize GroundingDINO if paths are provided
    groundingdino_model = None
    if hasattr(args, 'groundingdino_config') and os.path.isfile(resolve_path(args.groundingdino_config)):
        groundingdino_config = resolve_path(args.groundingdino_config)
        groundingdino_checkpoint = None
        
        if hasattr(args, 'groundingdino_checkpoint') and args.groundingdino_checkpoint:
            groundingdino_checkpoint = resolve_path(args.groundingdino_checkpoint)
            if not os.path.isfile(groundingdino_checkpoint):
                logger.warning(f"Local GroundingDINO checkpoint not found at: {groundingdino_checkpoint}")
                groundingdino_checkpoint = None
        
        # Use either local checkpoint or download from HuggingFace
        try:
            if groundingdino_checkpoint:
                # Use local checkpoint
                logger.info(f"Initializing GroundingDINO with local checkpoint: {groundingdino_checkpoint}")
                groundingdino_model = initialize_groundingdino(
                    groundingdino_config,
                    model_checkpoint_path=groundingdino_checkpoint,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif HF_AVAILABLE:
                # Use HuggingFace download
                logger.info(f"Initializing GroundingDINO from HuggingFace: {args.groundingdino_hf_repo}/{args.groundingdino_hf_filename}")
                groundingdino_model = initialize_groundingdino(
                    groundingdino_config,
                    hf_repo_id=args.groundingdino_hf_repo,
                    hf_filename=args.groundingdino_hf_filename,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                logger.error("Cannot load GroundingDINO model: no local checkpoint and huggingface_hub not available")
            
            if groundingdino_model:
                logger.info("GroundingDINO model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error initializing GroundingDINO: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"GroundingDINO config not found, skipping GroundingDINO initialization")

    # Process video with motion tracking
    if args.input_video:
        video_frames_folder = os.path.join(save_folder, 'video_frames')
        check_makedirs(video_frames_folder)
        total_time = process_video_with_motion_tracking(
            args.input_video, 
            model,
            transform.Compose([
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)
            ]),
            colors, 
            video_frames_folder, 
            args.video_output_path,
            alpha=args.alpha,
            motion_threshold=args.motion_threshold,
            static_weight=args.static_weight,
            grid_size=args.grid_size,
            groundingdino_model=groundingdino_model,
            spatial_tracking=args.spatial_tracking,
            revisit_threshold=args.revisit_threshold,
            overlap_threshold=args.overlap_threshold,
            revisit_discount=args.revisit_discount
        )
        
        pipeline_time = perf_counter() - pipeline_start
        logger.info(f"\nTotal pipeline time: {pipeline_time:.2f} seconds")
        logger.info(f"Processing time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()