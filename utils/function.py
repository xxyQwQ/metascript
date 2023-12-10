import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


class SquarePad(object):
    def __call__(self, image):
        _, width, height = image.shape
        target_size = max(width, height)
        pad_width = (target_size - width) // 2 + 10
        pad_height = (target_size - height) // 2 + 10
        return F.pad(image, (pad_width, pad_height, pad_width, pad_height), 'constant', 0)


class ColorReverse(object):
    def __call__(self, image):
        image = 1 - image
        image /= image.max()
        return image


def plot_sample(reference_image, template_image, script_image, result_image):
    def plot_grid(input):
        batch_size = input.shape[0]
        return 0.5 * make_grid(input.detach().cpu(), nrow=batch_size) + 0.5
    
    reference_count = reference_image.shape[1]
    reference_image = [plot_grid(reference_image[:, i, :, :].unsqueeze(1)) for i in range(reference_count)]
    template_image, script_image, result_image = plot_grid(template_image), plot_grid(script_image), plot_grid(result_image)
    sample_image = torch.cat([*reference_image, template_image, script_image, result_image], dim=1)
    return sample_image.numpy()
