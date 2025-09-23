# Advanced Video Segmentation with Motion Tracking and Object Detection

A comprehensive video segmentation pipeline that combines semantic segmentation with advanced motion tracking, spatial revisit detection, and object detection capabilities. This system is designed to provide intelligent analysis of video content with focus on urban scenes and autonomous driving scenarios.

## Demo

### Raw Video Segmentation
Transform raw video into segmented output with road, building, and vegetation classification.

**Input**: `basic_input.mp4`
- Raw video footage of urban/driving scene
- No preprocessing required

<video width="640" height="480" controls>
  <source src="https://github.com/Aashan47/Video-Segmentation/raw/main/basic_input.mp4" type="video/mp4">
  <a href="https://github.com/Aashan47/Video-Segmentation/raw/main/basic_input.mp4">Download basic_input.mp4</a>
</video>

**Output**: `output_basic_segmentation.mp4`
- Segmented video with color-coded regions
- Road (red), Building (blue), Vegetation (green)
- Overlay transparency configurable via `--alpha` parameter

<video width="640" height="480" controls>
  <source src="https://github.com/Aashan47/Video-Segmentation/raw/main/output_basic_segmentation.mp4" type="video/mp4">
  <a href="https://github.com/Aashan47/Video-Segmentation/raw/main/output_basic_segmentation.mp4">Download output_basic_segmentation.mp4</a>
</video>

```bash
# Basic segmentation demo
python tool/demo.py \
    --input_video basic_input.mp4 \
    --video_output_path output_basic_segmentation.mp4 \
    --config config/cityscapes/cityscapes_pspnet18_sd.yaml \
    --alpha 0.7
```

### Advanced Scene Detection
Intelligent processing with motion tracking and spatial revisit detection for optimized analysis.

**Input**: `input_Static.mp4`
- Video with static scenes and camera movement
- Contains revisited locations and stationary periods

<video width="640" height="480" controls>
  <source src="https://github.com/Aashan47/Video-Segmentation/raw/main/input_Static.mp4" type="video/mp4">
  <a href="https://github.com/Aashan47/Video-Segmentation/raw/main/input_Static.mp4">Download input_Static.mp4</a>
</video>

**Output**: `output-with-scene-detection.mp4`
- Motion-aware segmentation processing
- Static scene optimization
- Revisit detection and handling
- Enhanced statistics with scene analysis

<video width="640" height="480" controls>
  <source src="https://github.com/Aashan47/Video-Segmentation/raw/main/output-with-scene-detection.mp4" type="video/mp4">
  <a href="https://github.com/Aashan47/Video-Segmentation/raw/main/output-with-scene-detection.mp4">Download output-with-scene-detection.mp4</a>
</video>

```bash
# Advanced scene detection demo
python tool/demo.py \
    --input_video input_Static.mp4 \
    --video_output_path output-with-scene-detection.mp4 \
    --config config/cityscapes/cityscapes_pspnet18_sd.yaml \
    --motion_threshold 0.3 \
    --static_weight 0.0 \
    --spatial_tracking \
    --revisit_threshold 100 \
    --overlap_threshold 0.3 \
    --revisit_discount 0.0 \
    --alpha 0.7
```

**Key Differences in Advanced Output**:
- Reduced processing time through static scene detection
- More accurate label statistics with motion weighting
- Spatial revisit detection logs and visualization
- Enhanced CSV/JSON reports with scene analysis data

> **Note**: If videos don't play in your browser/viewer, you can:
> 1. Click the download links above to view locally
> 2. Visit the [repository files](https://github.com/Aashan47/Video-Segmentation) directly
> 3. Clone the repository and open videos with a media player

### Alternative: GIF Previews
If videos still don't work, consider converting key segments to GIF format:

```bash
# Convert videos to GIF for better compatibility (optional)
ffmpeg -i basic_input.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -t 10 basic_input_preview.gif
ffmpeg -i output_basic_segmentation.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -t 10 output_preview.gif
```

### Video File Structure
```
video-segmentation-main/
├── basic_input.mp4                    # Input demo video
├── output_basic_segmentation.mp4      # Basic segmentation result
├── input_Static.mp4                   # Static scene input video
├── output-with-scene-detection.mp4    # Advanced processing result
└── README.md                          # This file
```

---

## Features

### Core Capabilities
- **Semantic Segmentation**: PSPNet-based segmentation for road, building, and vegetation detection
- **Motion-Aware Processing**: CoTracker-based motion analysis to handle static vs dynamic scenes
- **Spatial Revisit Detection**: Advanced tracking to identify when camera revisits previously seen locations
- **Object Detection & Tracking**: GroundingDINO integration for enhanced object detection and instance tracking
- **Intelligent Label Statistics**: Motion-weighted and revisit-aware label percentage calculations

### Advanced Features
- **Look-ahead Motion Analysis**: Predictive motion scoring to optimize processing decisions
- **Static Scene Handling**: Intelligent weighting of static frames to avoid bias
- **Revisit Discount System**: Configurable discount factors for previously seen regions
- **Instance Tracking**: Unique object ID assignment and tracking across frames
- **Multi-modal Detection**: Combination of segmentation and object detection results

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for video processing)

### Dependencies
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy scipy scikit-image
pip install Pillow

# Motion tracking
pip install cotracker

# Object detection (GroundingDINO)
pip install groundingdino
pip install huggingface_hub

# Additional utilities
pip install argparse logging collections
```

### Model Setup
1. **PSPNet Model**: Download pretrained weights for semantic segmentation
2. **GroundingDINO**: Automatically downloaded from HuggingFace or provide local checkpoint
3. **CoTracker**: Downloaded via torch.hub

## Usage

### Basic Video Processing
```bash
python tool/demo.py \
    --input_video path/to/video.mp4 \
    --video_output_path output/segmented_video.mp4 \
    --config config/cityscapes/cityscapes_pspnet18_sd.yaml
```

### Advanced Motion-Aware Processing
```bash
python tool/demo.py \
    --input_video path/to/video.mp4 \
    --video_output_path output/segmented_video.mp4 \
    --motion_threshold 0.3 \
    --static_weight 0.0 \
    --grid_size 10 \
    --alpha 0.7
```

### Spatial Tracking with Revisit Detection
```bash
python tool/demo.py \
    --input_video path/to/video.mp4 \
    --video_output_path output/segmented_video.mp4 \
    --spatial_tracking \
    --revisit_threshold 100 \
    --overlap_threshold 0.3 \
    --revisit_discount 0.0
```

### Full Feature Pipeline
```bash
python tool/demo.py \
    --input_video path/to/video.mp4 \
    --video_output_path output/segmented_video.mp4 \
    --config config/cityscapes/cityscapes_pspnet18_sd.yaml \
    --motion_threshold 0.3 \
    --static_weight 0.0 \
    --spatial_tracking \
    --revisit_threshold 100 \
    --overlap_threshold 0.3 \
    --revisit_discount 0.0 \
    --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --alpha 0.7 \
    --rotate 0
```

## Configuration Parameters

### Motion Tracking Parameters
- `--motion_threshold`: Threshold for motion detection (0.0-1.0, default: 0.3)
- `--static_weight`: Weight applied to static scenes (0.0-1.0, default: 0.0)
- `--grid_size`: Grid size for CoTracker point tracking (default: 10)

### Spatial Tracking Parameters
- `--spatial_tracking`: Enable spatial tracking for revisit detection
- `--revisit_threshold`: Minimum frames between potential revisits (default: 100)
- `--overlap_threshold`: Minimum overlap ratio for revisit detection (default: 0.3)
- `--revisit_discount`: Discount factor for labels in revisited regions (0.0-1.0, default: 0.0)

### GroundingDINO Parameters
- `--groundingdino_config`: Path to GroundingDINO configuration file
- `--groundingdino_checkpoint`: Path to local checkpoint (optional)
- `--groundingdino_hf_repo`: HuggingFace repository ID (default: "ShilongLiu/GroundingDINO")
- `--groundingdino_hf_filename`: HF checkpoint filename (default: "groundingdino_swint_ogc.pth")

### Video Processing Parameters
- `--input_video`: Path to input video file
- `--video_output_path`: Output video path (default: 'segmentation_output.mp4')
- `--alpha`: Transparency for segmentation overlay (0.0-1.0, default: 0.7)
- `--rotate`: Rotation angle (0, 90, 180, 270 degrees, default: 0)

## Architecture Overview

### Core Components

#### 1. Motion Tracking System
- **CoTracker Integration**: Real-time point tracking across video frames
- **Motion Score Calculation**: Quantitative motion analysis based on point displacement
- **Look-ahead Analysis**: Predictive motion scoring to optimize processing decisions
- **Static Scene Detection**: Automatic identification of stationary camera periods

#### 2. Spatial Tracking System
- **Feature-based Matching**: SIFT feature detection and matching
- **Homography Estimation**: RANSAC-based transformation estimation
- **Revisit Detection**: Identification of previously seen spatial regions
- **Keyframe Management**: Efficient storage and retrieval of spatial reference frames

#### 3. Instance Tracking System
- **Object ID Assignment**: Unique identifier tracking across frames
- **IoU-based Matching**: Intersection over Union matching for object continuity
- **Multi-class Support**: Simultaneous tracking of different object classes
- **Disappearance Handling**: Robust handling of temporarily occluded objects

#### 4. Enhanced Object Detection
- **GroundingDINO Integration**: Text-prompt based object detection
- **Class-specific Prompts**: Tailored detection prompts for road, building, vegetation
- **Segmentation Enhancement**: Fusion of detection results with segmentation masks
- **Confidence Filtering**: Quality-based filtering of detection results

### Processing Pipeline

1. **Video Input & Preprocessing**
   - Frame extraction and rotation handling
   - Resolution normalization
   - Buffer management for look-ahead analysis

2. **Motion Analysis**
   - CoTracker point tracking
   - Motion score calculation
   - Static scene identification
   - Look-ahead prediction

3. **Spatial Analysis** (Optional)
   - Feature extraction and matching
   - Homography estimation
   - Revisit detection and mapping
   - Keyframe management

4. **Semantic Segmentation**
   - PSPNet-based pixel classification
   - Multi-class segmentation (road, building, vegetation)
   - Confidence-based filtering

5. **Object Detection** (Optional)
   - GroundingDINO text-prompt detection
   - Class-specific object identification
   - Instance tracking and ID assignment
   - Detection-segmentation fusion

6. **Statistical Analysis**
   - Motion-weighted label counting
   - Revisit-aware statistics
   - Instance-based percentages
   - Export to CSV/JSON formats

## Output Files

### Video Output
- **Segmented Video**: MP4 file with overlay visualization
- **Frame Sequence**: Individual processed frames (PNG format)

### Statistical Output
- **CSV Report**: Label percentages and statistics
- **JSON Report**: Detailed analysis results with metadata

### Log Output
- **Processing Logs**: Detailed execution information
- **Performance Metrics**: Timing and efficiency statistics
- **Detection Results**: Object detection and tracking logs

## Supported Classes

The system currently supports three primary classes optimized for urban/driving scenarios:

- **Road (Class 0)**: Streets, paths, highways
- **Building (Class 2)**: Architecture, structures, houses, skyscrapers  
- **Vegetation (Class 8)**: Trees, plants, grass, forest

## Advanced Use Cases

### 1. Autonomous Driving Analysis
```bash
# Optimize for driving scenarios with high motion sensitivity
python tool/demo.py \
    --input_video driving_video.mp4 \
    --motion_threshold 0.2 \
    --spatial_tracking \
    --revisit_discount 0.1
```

### 2. Surveillance Video Analysis
```bash
# Handle static cameras with revisit detection
python tool/demo.py \
    --input_video surveillance.mp4 \
    --static_weight 0.5 \
    --spatial_tracking \
    --revisit_threshold 50
```

### 3. Drone/Aerial Video Processing
```bash
# Account for camera movement and spatial revisits
python tool/demo.py \
    --input_video aerial_video.mp4 \
    --motion_threshold 0.4 \
    --spatial_tracking \
    --overlap_threshold 0.4
```

## Performance Optimization

### GPU Acceleration
- CUDA support for all major components
- Optimized tensor operations
- Batch processing where applicable

### Memory Management
- Efficient frame buffering
- Selective processing based on motion analysis
- Garbage collection optimization

### Processing Speed
- Look-ahead optimization reduces redundant processing
- Static scene skipping improves efficiency
- Multi-threaded where possible

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce grid_size parameter
   - Process shorter video segments
   - Lower input resolution

2. **CoTracker Initialization Failed**
   - Ensure proper torch.hub access
   - Check internet connectivity
   - Verify CUDA compatibility

3. **GroundingDINO Model Loading**t combines multiple state-of-the-art models and techniques. Contributions are welcome in the following areas:
   - Verify HuggingFace access
   - Check configuration file paths
   - Ensure sufficient disk space- Improved motion analysis algorithms
d spatial tracking methods
### Performance Issuesions
- **Slow Processing**: Increase motion_threshold to skip more static frames
- **High Memory Usage**: Reduce look_ahead_size or grid_size
- **Poor Detection**: Adjust box_threshold and text_threshold values

## Contributingf you use this work in your research, please cite the relevant papers:

## Citation

If you use this work in your research, please cite the relevant papers: Track Together},
d others},
```bibtexXiv preprint},
@article{cotracker2024, year={2024}
  title={CoTracker: It is Better to Track Together},
  author={Karaev, Nikita and others},
  journal={arXiv preprint},roundingdino2023,
  year={2024}  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
}
  journal={arXiv preprint},
@article{groundingdino2023,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},}
  author={Liu, Shilong and others},
  journal={arXiv preprint},
  year={2023}
}
```This project builds upon multiple open-source components. Please refer to individual component licenses for specific terms.












- OpenCV and scikit-image communities for computer vision tools- PSPNet authors for semantic segmentation foundation- GroundingDINO team for object detection framework- CoTracker team for motion tracking capabilities## AcknowledgmentsThis project builds upon multiple open-source components. Please refer to individual component licenses for specific terms.## License
## Acknowledgments

- CoTracker team for motion tracking capabilities
- GroundingDINO team for object detection framework
- PSPNet authors for semantic segmentation foundation
- OpenCV and scikit-image communities for computer vision tools
