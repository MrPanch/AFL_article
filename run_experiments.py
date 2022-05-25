from train import train


def run_experiments():
    crop_sizes = [256]
    upscale_factors = [4]
    num_epochs = [100]
    AFLs = ['no_AFL', 'G_log', 'G_no_log', 'L_log', 'L_no_log']
    models = ['SRGAN', 'ESRGAN']
    datasets = ['DIV2K', 'BPUI', 'BUSI']

    for AFL in AFLs:
        for model in models:
            train(crop_sizes[0], upscale_factors[0], num_epochs[0], AFL, model, 'BUSI')


if __name__ == '__main__':
    run_experiments()
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