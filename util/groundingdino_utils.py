"""
Patched versions of GroundingDINO inference utilities to fix PIL Image compatibility issues.
"""
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert

def patched_predict(
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

def patched_annotate(image_pil, boxes, labels, colors=None, output_path=None):
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