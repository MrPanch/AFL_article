import argparse
import os
import json
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from test_image import test_model
from models.generator import Generator, eGenerator
from models.discriminator import Discriminator, eDiscriminator
from piq import ssim, psnr, fsim, brisque, gmsd, mdsi

# parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
# parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
#                     help='super resolution upscale factor')
# parser.add_argument('--num_epochs', default=150, type=int, help='train epoch number')
# parser.add_argument('--AFL', default='no_AFL', choices=['no_AFL', 'G_log', 'G_no_log', 'L_log', 'L_no_log'],
#                     help='using of AFL layer')
# parser.add_argument('--model_type', default='ESRGAN', choices=['SRGAN', 'ESRGAN'],
#                     help='model type')
# parser.add_argument('--dataset', choices=['DIV2K', 'MSTAR', 'BPUI', 'BUSI'],
#                     help='model type')

def train(CROP_SIZE, UPSCALE_FACTOR, NUM_EPOCHS, AFL_type, model_type, dataset_type):
    print("\n ### RUNNING MODEL WITH PARAMS### \n", CROP_SIZE, UPSCALE_FACTOR, NUM_EPOCHS, AFL_type, model_type, dataset_type)
    # Parse parameters
    # opt = parser.parse_args()
    #
    # CROP_SIZE = opt.crop_size
    # UPSCALE_FACTOR = opt.upscale_factor
    # NUM_EPOCHS = opt.num_epochs
    # AFL_type = opt.AFL

    if dataset_type == 'DIV2K':
        train_path = 'data/DIV2K_train_HR'
        val_path = 'data/DIV2K_valid_HR'
    elif dataset_type == 'MSTAR':
        train_path = 'data/MSTAR-10_original/train'
        val_path = 'data/MSTAR-10_original/test'
    elif dataset_type == 'BPUI':
        train_path = 'data/nerve_segmentation/train'
        val_path = 'data/nerve_segmentation/test'
    elif dataset_type == 'BUSI':
        train_path = 'data/Dataset_BUSI_with_GT/train'
        val_path = 'data/Dataset_BUSI_with_GT/test'
    else:
        RuntimeError("Wrong dataset option")

    # Create train and val datasets
    train_set = TrainDatasetFromFolder(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(val_path, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=2, shuffle=False)

    # Create models
    if model_type is "SRGAN":
        netG = Generator(UPSCALE_FACTOR, CROP_SIZE, AFL_type)
        netD = Discriminator()
    else:
        netG = eGenerator(UPSCALE_FACTOR, CROP_SIZE, AFL_type)
        netD = eDiscriminator(CROP_SIZE)
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # Init optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=0.001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.001)

    # Init scores
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [],
               'fsim': [], 'brisque': [], 'gmsd': [], 'mdsi': []}
    psnr_max, ssim_max, fsim_max = -1, -1, -1
    brisque_min, gmsd_min, mdsi_min = np.Inf, np.Inf, np.Inf

    # Set output paths
    score_path_root = f'runs/{AFL_type}_{UPSCALE_FACTOR}/'
    if not os.path.exists(score_path_root):
        os.makedirs(score_path_root)
    exp_number = len(os.listdir(score_path_root))
    score_path = f"{score_path_root}/{exp_number}/"
    if not os.path.exists(score_path):
        os.makedirs(score_path)
        os.makedirs(os.path.join(score_path, "psnr"))
        os.makedirs(os.path.join(score_path, "ssim"))

    # Save hype parameters to file
    with open(f'{score_path}/hype_parameters.txt', 'w') as f:
        params = {
            'crop_size': CROP_SIZE,
            'UPSCALE_FACTOR': UPSCALE_FACTOR,
                 'NUM_EPOCHS': NUM_EPOCHS,
                 'AFL_type': AFL_type,
                 'model_type': model_type,
                 'dataset_type': dataset_type
                }
        json.dump(params, f, indent=2)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            # g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################

            if torch.cuda.is_available():
                real_img = target.cuda()
                z = data.cuda()
            else:
                real_img = target
                z = data

            # Generate fake image
            fake_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
            break
        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'psnr': 0, 'ssim': 0, 'fsim': 0, 'brisque': 0, 'gmsd': 0, 'mdsi': 0}
            current_batch = 0
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
                current_batch += batch_size
                valing_results['ssim'] += ssim(sr, hr).item() * batch_size
                valing_results['psnr'] += psnr(sr, hr).item() * batch_size
                valing_results['fsim'] += fsim(sr, hr).item() * batch_size
                valing_results['brisque'] += brisque(sr).item() * batch_size
                valing_results['gmsd'] += gmsd(sr, hr).item() * batch_size
                valing_results['mdsi'] += mdsi(sr, hr).item() * batch_size



                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f FSIM: %.4f BRISUQE: %.4f GMSD: %.4f MDSI: %.4f' % (
                        valing_results['psnr'] / current_batch,
                        valing_results['ssim'] / current_batch,
                        valing_results['fsim'] / current_batch,
                        valing_results['brisque'] / current_batch,
                        valing_results['gmsd'] / current_batch,
                        valing_results['mdsi'] / current_batch))

        # save model parameters
        if valing_results['psnr'] > psnr_max:
            torch.save(netG.state_dict(), f'{score_path}/psnr/netG_{epoch}.pth')
            torch.save(netD.state_dict(), f'{score_path}/psnr/netD.pth')
            psnr_max = valing_results['psnr']
        if valing_results['ssim'] > ssim_max:
            torch.save(netG.state_dict(), f'{score_path}/ssim/netG_{epoch}.pth')
            torch.save(netD.state_dict(), f'{score_path}/ssim/netD.pth')
            ssim_max = valing_results['ssim']
        # if valing_results['fsim'] > fsim_max:
        #     torch.save(netG.state_dict(), f'{score_path}/fsim_max/netG_{epoch}.pth')
        #     torch.save(netD.state_dict(), f'{score_path}/fsim_max/netD.pth')
        #     fsim_max = valing_results['fsim']

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'] / current_batch)
        results['ssim'].append(valing_results['ssim'] / current_batch)
        results['fsim'].append(valing_results['fsim'] / current_batch)
        results['brisque'].append(valing_results['brisque'] / current_batch)
        results['gmsd'].append(valing_results['gmsd'] / current_batch)
        results['mdsi'].append(valing_results['mdsi'] / current_batch)


        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim'],
                  'FSIM': results['fsim'], 'BRISQUE': results['brisque'], 'GMSD': results['gmsd'],
                  'MDSI': results['mdsi']},
            index=range(1, epoch + 1))
        data_frame.to_csv(f'{score_path}/train_results.csv', index_label='Epoch')

        if epoch % 2 == 0:
            print("\n Running test on image \n")
            if dataset_type == 'BUSI':
                test_model(netG, f'data/test_data/normal (117).png', score_path, epoch)
            else:
                raise RuntimeError
