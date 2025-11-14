import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

from model.model import TwinLiteNetPlus, TwinLiteNetPlus_Lightweight, TwinLiteNetPlus_V3, TwinLiteNetPlus_Vmamba2, TwinLiteNetPlus_ConvNextV2
from demoDataset import LoadImages, LoadStreams
from tqdm import tqdm
import time
from utils import netParams
from utils import count_flops_and_params

def detect(args):

    device = "cuda:0"
    # half = True
    half = False
    model = TwinLiteNetPlus_ConvNextV2(args)
    model.eval().cuda()
    
    flops, params = count_flops_and_params(model)
    print(f'Total network GFlops: {flops / 1e9}')
    print(f'Total network parameters: {params}')

    if half:
        model.half()  # to FP16

    # Set Dataloader
    if args.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=args.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)
        bs = 1  # batch_size
    # Run inference
    t0 = time.time()

    # model.load_state_dict(torch.load(args.weight))

    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device != 'cpu' else None  # run once
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.half().cuda() if half else img.cuda()


        # _, _, height, width = img.shape
        # Inference
        # t1 = time.time()
        da_seg_out,ll_seg_out= model(img)
        # t2 = time.time()

    t1 = time.time()
    print(f"processed {len(dataset)} images in {t1 - t0} seconds and batch size {bs}, throughput {len(dataset) / (t1 - t0)} imgs/s")

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--config', type=str, help='Model configuration')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)