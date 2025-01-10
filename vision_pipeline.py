
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# SAM2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Cutie
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from camera import ZEDCamera


class SAM:
    """
    Segment Anything: generates masks of distinct objects in an RGB image.
    """

    def __init___(self):
        sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"), apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

    def generate(self, image):
        mask_dict = self.mask_generator.generate(image)
        masks = [mask_dict[i]["segmentation"] for i in range(len(mask_dict))]
        return masks


class Cutie:
    """
    Pipeline used for tracking a mask in a video.
    """

    def __init__(self, init_mask, init_image):
               
        # obtain the Cutie model with default parameters -- skipping hydra configuration
        cutie = get_default_model()
        self.processor = InferenceCore(cutie, cfg=cutie.cfg)
        self.processor.max_internal_size = 480

        # Generate mask
        objects = np.unique(np.array(init_mask))
        objects = objects[objects != 0].tolist()
        mask = torch.from_numpy(np.array(init_mask)).cuda()

        image = to_tensor(init_image).cuda().float()

        # Generate output for first step
        self.processor.step(image, mask, objects=objects)


    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def generate(self, image):
        """
        Generates prediction

        Input:
            image: Image.PIL as RGB image
        
        Output:
            mask: Image.PIL of predicted mask
        """

        # load the image as RGB; normalization is done within the model
        image = to_tensor(image).cuda().float()
        
        # prediction
        output_prob = self.processor.step(image)

        # convert output probabilities to an object mask
        mask = self.processor.output_prob_to_mask(output_prob)
        mask = mask.cpu().numpy().astype(np.uint8)

        return mask


if __name__ == "__main__":

    z = ZEDCamera()
    image = z.capture_image("rgb")

    # sam = SAM()
    # masks = sam.generate(image)
    # for i, M in enumerate(masks):
    #     m = M.astype("uint8")
    #     m = m * 255
    #     Image.fromarray(m).convert("RGB").save(f"{i}.png")

    # Load mask
    mask = np.array(Image.open("13.png").convert("L"))
    mask_tracker = Cutie(mask, image)

    # matplotlib display
    plt.ion()
    fig, ax = plt.subplots()
    img_display = ax.imshow(image)
    plt.title("Live Camera Feed")
    plt.axis("off")

    # Display mask
    try:
        while True:
            rgb = z.capture_image("rgb")
            mask_track = mask_tracker.generate(rgb)
            img_display.set_data(mask_track)
            plt.draw()             # Redraw the current figure
            plt.pause(0.01)        # Brief pause to process GUI events
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
        exit()