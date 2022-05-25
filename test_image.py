import argparse
import time
import skimage.io
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from models.generator import Generator, eGenerator
import os
#
# parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
# parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='default_no_AFL/netG_epoch_4_100.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()
#
# UPSCALE_FACTOR = opt.upscale_factor
# TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = opt.image_name
# MODEL_NAME = opt.model_name
#
# model = Generator(UPSCALE_FACTOR, 100).eval()
# if TEST_MODE:
#     model.cuda()
#     model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
# else:
#     model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
#
# image = Image.open(IMAGE_NAME).convert('RGB')
# image = image.resize((25, 25))
# image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
# if TEST_MODE:
#     image = image.cuda()


def test_model(model, path2img, root_path, epoch):
    image = Image.open(path2img).convert('RGB')
    image = CenterCrop(model.crop_size // model.scale_factor)(image)
    with torch.no_grad():
        image = Variable(ToTensor()(image)).unsqueeze(0).cuda()
        out = model(image)
        out_img = ToPILImage()(out[0].data.cpu())

    save_dir = os.path.join(root_path, 'test_images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    out_img.save(f'{save_dir}/{epoch}.jpg')
#
# start = time.clock()
# out = model(image)
# elapsed = (time.clock() - start)
# print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('./data/test/' + str(UPSCALE_FACTOR) + '_' + 'no_AFL_1.jpg')
