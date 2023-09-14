import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_pil_image, InterpolationMode

from pvit.model.region_clip import get_region_clip_transforms

target_size = 224
clip_image_transform = transforms.Compose([
    transforms.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


region_clip_image_transform = get_region_clip_transforms()