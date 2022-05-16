import torch

# Depth Deep Learning Model
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def get_depth_from_color(color_img):
    """ Get depth image from midas DL model """
    input_batch = transform(color_img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=color_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    output = (output/40*255).astype(int) # Here is to scale the range of value to approach 255
    return output

def masking_uninterested(depth_image, threshold_min = 60, threshold_max = 170):
    """ Masking the sky (and too closed object) by threshold based on empirical testing """
    threshold_min = 65
    threshold_max = 170
    depth_image[depth_image < threshold_min] = 0
    depth_image[depth_image > threshold_max] = 0
    return depth_image