import os
import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn
import yaml
import math
from copy import deepcopy
from argparse import ArgumentParser

from model.model import SingleLiteNetPlus
from loss import SigleLoss
from utils import train, val_one, netParams, save_checkpoint, poly_lr_scheduler
import BDD100K

class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # Exponential decay ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA model parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def train_net(args, hyp):
    """Train the neural network model with given arguments and hyperparameters"""
    use_ema = args.ema
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    
    model = SingleLiteNetPlus(args)

    # Load pretrained weights if available for this config
    pretrained_path = os.path.join('./pretrained', f"{args.config}.pth")
    if os.path.isfile(pretrained_path):
        print(f"=> Loading pretrained weights from '{pretrained_path}'")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            # If checkpoint is a dict with 'state_dict' key
            state_dict = state_dict['state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Filter out keys that are not in the current model and handle decoder naming differences
        model_state = model.state_dict()
        filtered_state_dict = {}
        
        def map_decoder_key(key, task):
            """Map decoder keys from 2-head model to single-head model"""
            if task == "DA" and '_da' in key:
                # Replace _da with nothing, but handle the full path
                return key.replace('_da', '')
            elif task == "LL" and '_ll' in key:
                # Replace _ll with nothing, but handle the full path  
                return key.replace('_ll', '')
            return key
        
        for k, v in new_state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                # Direct match
                filtered_state_dict[k] = v
            else:
                # Try mapping decoder keys
                mapped_key = map_decoder_key(k, args.task)
                if mapped_key in model_state and v.size() == model_state[mapped_key].size():
                    filtered_state_dict[mapped_key] = v
        
        missing_keys = set(model_state.keys()) - set(filtered_state_dict.keys())
        unexpected_keys = set(new_state_dict.keys()) - set(filtered_state_dict.keys())
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"=> Pretrained weights loaded. ({len(filtered_state_dict)} matched keys, {len(missing_keys)} missing, {len(unexpected_keys)} unexpected)")
        if missing_keys:
            print(f"=> Missing keys: {list(missing_keys)[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"=> Unexpected keys: {list(unexpected_keys)[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    
    os.makedirs(args.savedir, exist_ok=True)  # Ensure save directory exists
    
    trainLoader = torch.utils.data.DataLoader(
        BDD100K.DatasetOneTask(hyp, valid=False,task=args.task),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    valLoader = torch.utils.data.DataLoader(
        BDD100K.DatasetOneTask(hyp, valid=True,task=args.task),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
    
    print(f'Total network parameters: {netParams(model)}')
    
    criteria = SigleLoss(hyp,task=args.task)
    start_epoch = 0
    lr = hyp['lr']
    
    # For fine-tuning, use different learning rates for different parts
    if os.path.isfile(pretrained_path):
        print("=> Fine-tuning mode: Using different learning rates for encoder vs decoder")
        # Lower learning rate for pretrained encoder, higher for decoder
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name or 'caam' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.2},  # 5x lower for encoder
            {'params': decoder_params, 'lr': lr}         # Normal lr for decoder
        ], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
    
    ema = ModelEMA(model) if use_ema else None
    
    # Resume training from checkpoint
    if args.resume and os.path.isfile(args.resume):
        if args.resume.endswith(".tar"):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if use_ema:
                ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                ema.updates = checkpoint['updates']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No valid checkpoint found at '{args.resume}'")
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.max_epochs):
        model_file_name = os.path.join(args.savedir, f'model_{epoch}.pth')
        
        # Warmup for fine-tuning
        if os.path.isfile(pretrained_path) and epoch < 5:
            # Warmup: gradually increase learning rate
            warmup_factor = (epoch + 1) / 5.0
            for param_group in optimizer.param_groups:
                if 'encoder' in str(param_group) or 'caam' in str(param_group):
                    param_group['lr'] = hyp['lr'] * 0.1 * warmup_factor
                else:
                    param_group['lr'] = hyp['lr'] * warmup_factor
        else:
            poly_lr_scheduler(args, hyp, optimizer, epoch)
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print(f"Learning rate: {lr}")
        
        model.train()
        ema = train(args, trainLoader, model, criteria, optimizer, epoch, scaler, args.verbose, ema if use_ema else None)
        ema = ema[0]
        
        model.eval()
        segment_results = val_one(valLoader, ema.ema if use_ema else model, args=args)
        
        print(f"Driving Area Segment: mIOU({segment_results[2]:.3f})") if args.task=="DA" else \
            print(f"Lane Line Segment: Acc({segment_results[0]:.3f}) IOU({segment_results[1]:.3f})")
        
        torch.save(ema.ema.state_dict(), model_file_name) if use_ema else torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.ema.state_dict() if use_ema else None,
            'updates': ema.updates if use_ema else None,
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, os.path.join(args.savedir, 'checkpoint.pth.tar'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savedir', default='./testv3', help='Directory to save the results')
    parser.add_argument('--task', type=str, choices=["DA", "LL"], default=None, help="DA for drivable area, LL for lane line")
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML')
    parser.add_argument('--resume', type=str, default='', help='Resume training from a checkpoint')
    parser.add_argument('--config', default='nano', help='Model configuration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--ema', action='store_true', help='Use Exponential Moving Average (EMA)')
    args = parser.parse_args()
    
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters
    
    train_net(args, hyp.copy())
