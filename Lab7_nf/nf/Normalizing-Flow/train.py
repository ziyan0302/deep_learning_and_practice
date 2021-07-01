import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from dataset import ICLEVRLoader, Lab7_Dataset
import util
import wandb
from models import Glow
from tqdm import tqdm
from evaluator import EvaluationModel 
from test_util import get_test_conditions, save_image
import copy



def main(args):
    # Set up main device and scale batch size
    wandb.init(project='dlp-lab7-task1-nf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainset = ICLEVRLoader(mode="train")
    print('trainset: ', trainset)
    datasetDir_path = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1'
    datasetImgDir_path = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1/images'
    testset = Lab7_Dataset(img_path = datasetImgDir_path, json_path = os.path.join(datasetDir_path,'test.json'))
    print('testset: ', testset)


    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps)
    net = net.to(device)
    wandb.watch(net)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net, args.gpu_ids)
    #     cudnn.benchmark = args.benchmark

    start_epoch = 1
    # if args.resume:
    #     # Load checkpoint.
    #     print('Resuming from checkpoint at ckpts/best.pth.tar...')
    #     assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('ckpts/best.pth.tar')
    #     net.load_state_dict(checkpoint['net'])
    #     global best_loss
    #     global global_step
    #     best_loss = checkpoint['test_loss']
    #     start_epoch = checkpoint['epoch']
    #     global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    train(args.num_epochs, net, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
    # for epoch in range(start_epoch, start_epoch + args.num_epochs):
    #     train(epoch, net, trainloader, device, optimizer, scheduler,
    #           loss_fn, args.max_grad_norm)
    #     test_1(net, testset, device)


@torch.enable_grad()
def train(epochs, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step

    net.train()
    loss_meter = util.AverageMeter()
    evaluator = EvaluationModel()
    test_conditions = get_test_conditions(os.path.join('test.json')).to(device)
    new_test_conditions = get_test_conditions(os.path.join('new_test.json')).to(device)
    best_score = 0
    new_best_score = 0
    
    for epoch in range(1, epochs+1):
        print('\nEpoch: ', epoch)
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for x, cond_x in trainloader:
                x , cond_x= x.to(device, dtype=torch.float), cond_x.to(device, dtype=torch.float)
                optimizer.zero_grad()
                z, sldj = net(x, cond_x, reverse=False)
                loss = loss_fn(z, sldj)
                wandb.log({'loss':loss})
                # print('loss: ',loss)
                loss_meter.update(loss.item(), x.size(0))
                # wandb.log({'loss_meter',loss_meter})
                loss.backward()
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                # scheduler.step(global_step)

                progress_bar.set_postfix(nll=loss_meter.avg,
                                        bpd=util.bits_per_dim(x, loss_meter.avg),
                                        lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(x.size(0))
                global_step += x.size(0)
        
        net.eval()
        with torch.no_grad():
            gen_imgs = sample(net,test_conditions, device)
        score = evaluator.eval(gen_imgs, test_conditions)
        wandb.log({'score': score})
        if score > best_score:
                best_score = score
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(best_model_wts, os.path.join('weightings/test', f'epoch{epoch}_score{score:.2f}.pt'))
        
        with torch.no_grad():
            new_gen_imgs = sample(net,new_test_conditions, device)
        new_score = evaluator.eval(new_gen_imgs, new_test_conditions)
        wandb.log({'new_score': new_score})
        if new_score > new_best_score:
                new_best_score = score
                new_best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(best_model_wts, os.path.join('weightings/new_test', f'epoch{epoch}_score{score:.2f}.pt'))
        save_image(gen_imgs, os.path.join('results/test', f'epoch{epoch}.png'), nrow=8, normalize=True)
        save_image(new_gen_imgs, os.path.join('results/new_test', f'epoch{epoch}.png'), nrow=8, normalize=True)
        

@torch.no_grad()
def sample(net, condition, device, sigma=0.6):
    B = len(condition)
    z = torch.randn((B, 3, 64, 64), dtype=torch.float32, device=device) * sigma
    x, _ = net(z, condition, reverse=True)
    x = torch.sigmoid(x)

    return x

'''
@torch.no_grad()

def test(epoch, net, testloader, device, loss_fn, mode='color'):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()

    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, x_cond in testloader:
            x, x_cond = x.to(device), x_cond.to(device)
            z, sldj = net(x, x_cond, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg
        wandb.log({'best_loss', best_loss})
    origin_img, gray_img = next(iter(testloader))
    B = gray_img.shape[0]
    # Save samples and data
    images = sample(net, gray_img, device)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('ref_pics', exist_ok=True)
    if mode == 'sketch':
        gray_img = (~gray_img.type(torch.bool)).type(torch.float)
    images_concat = torchvision.utils.make_grid(images, nrow=int(B ** 0.5), padding=2, pad_value=255)
    origin_concat = torchvision.utils.make_grid(origin_img, nrow=int(B ** 0.5), padding=2, pad_value=255)
    gray_concat = torchvision.utils.make_grid(gray_img, nrow=int(B ** 0.5), padding=2, pad_value=255)

    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))
    torchvision.utils.save_image(origin_concat, 'ref_pics/origin_{}.png'.format(epoch))
    torchvision.utils.save_image(gray_concat, 'ref_pics/gray_{}.png'.format(epoch))

def test_1(net, testset, device):
    net.eval()
    print('test_1')
    print('testset: ', testset)
    for i in range(10):
        print('ok1')
        gray_img = testset[i]
        gen_imgs = sample(net,gray_img,device)
        images = sample(net, gray_img, device)
        print('ok2')
'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=4, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=6, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=500, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    best_loss = float('inf')
    global_step = 0

    main(parser.parse_args())