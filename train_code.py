import os
import torch
import argparse
from model.code import CodeModel, CondARModel
from dataset import CodeDataset
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter



def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Initialize dataset loader
    dataset = CodeDataset(datapath=args.input, maxlen=args.seqlen, names_path=args.names_path, res=args.res, voxel_path=args.voxel_path, cache=args.cache) 
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=5)
    # Initialize vertex model
    if args.use_voxels:
        model = CondARModel(config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=args.seqlen,
        classes=args.code,)
        model_name = 'code_voxel'
    else:
        model = CodeModel(
            config={
                'hidden_dim': 512,
                'embed_dim': 256, 
                'num_layers': 8,
                'num_heads': 8,
                'dropout_rate': 0.1
            },
            max_len=args.seqlen,
            classes=args.code,
        )
        model_name = 'code'
    model = model.to(device).train()
    
    # Initialize optimizer
    network_parameters = list(model.parameters()) 
    optimizer = torch.optim.Adam(network_parameters, lr=1e-3)
   
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print(f'Start training {model_name}...')

    for epoch in range(800): 
        print(epoch)

        for batch in dataloader:
            if args.use_voxels:
                code = batch['code']
                voxels = batch['voxels']
                voxels = voxels.to(device)
                
            else:
                code = batch
            code = code.to(device)

            # Pass through vertex prediction module 
            arglist = [code[:,:-1]]
            if args.use_voxels:
                arglist.append(voxels)
            logits = model(*arglist)

            c_pred = logits.reshape(-1, logits.shape[-1]) 
            c_target = code.reshape(-1)
            code_loss = F.cross_entropy(c_pred, c_target)
           
            total_loss = code_loss

            # logging
            if iters % 20 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters)

            # Backprop 
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(network_parameters, max_norm=1.0)  # clip gradient
            optimizer.step()
            iters += 1

        writer.flush()

        # save model after n epoch
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output,f'{model_name}_epoch_'+str(epoch+1)+'.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--code", type=int, required=True)
    parser.add_argument("--use_voxels", action='store_true')
    parser.add_argument("--voxel_path", type=str)
    parser.add_argument("--names_path", type=str)
    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--no_cache", action='store_false', dest='cache')
    parser.add_argument("--save_interval", type=int, default=10)

    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)
