import pdb
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate(model, dataset, device, filename):
    pdb.set_trace()
    image, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, mask = model(image.to(device))
    output = output.to(torch.device('cpu'))
    mask = mask.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

    
