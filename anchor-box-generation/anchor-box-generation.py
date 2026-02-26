def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    box = []

    for y in range(feature_size):
        cy = (y + 0.5) * stride
        
        for x in range(feature_size):
            cx = (x + 0.5) * stride

            for s in scales:
                for ratio in aspect_ratios:
                    w = s * (ratio ** (1 / 2))
                    h = s / (ratio ** (1 / 2))

                    box.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    return box