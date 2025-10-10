import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from argparse import ArgumentParser
from pathlib import Path

from model.model import TwinLiteNetPlus
from utils import val, netParams
from loss import TotalLoss
import BDD100K


def validation(args):
    """
    Perform model validation on the BDD100K dataset.
    :param args: Parsed command-line arguments.
    """
    
    # Initialize model
    model = TwinLiteNetPlus(args)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
    
    # Load hyperparameters from YAML file
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    
    # Create validation data loader
    valLoader = torch.utils.data.DataLoader(
        BDD100K.Dataset(hyp, valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Print model parameter count
    print(f'Total network parameters: {netParams(model)}')
    
    # Load pretrained weights
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    
    # Perform validation
    da_segment_results, ll_segment_results = val(valLoader, model, args.half, args=args)
    
    # Print results
    print(f"Driving Area Segment: mIOU({da_segment_results[2]:.3f})")
    print(f"Lane Line Segment: Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f})")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default="pretrained/large.pth", help='Path to model weights')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large", "vmamba-tiny", "vmamba-tiny-swiglu"], help='Model configuration')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML file')
    parser.add_argument('--half', action='store_true', help='Use half precision for inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Parse arguments and run validation
    validation(parser.parse_args())
