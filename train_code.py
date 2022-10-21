import os
import torch
import argparse
from model.code import CodeModel, CondARModel
from model.encoder import CMDEncoder, EXTEncoder, PARAMEncoder
from dataset import CodeDataset
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter



def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Initialize dataset loader
    dataset = CodeDataset(datapath=args.input, names_path=args.names_path, res=args.res, voxel_path=args.voxel_path, cache=args.cache, splits_file=args.splits_file, mode='train') 
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
            'dropout_rate': args.dropout
        },
        encoder_config={
            'hidden_dim': 256,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': args.dropout
        },
        max_len=args.seqlen,
        classes=128 if args.continuous_decode else args.code,
        use_transformer_encoder=args.encoder, continuous_decode=args.continuous_decode)
        model_name = 'code_voxel'
    else:
        model = CodeModel(
            config={
                'hidden_dim': 512,
                'embed_dim': 256, 
                'num_layers': 8,
                'num_heads': 8,
                'dropout_rate': args.dropout
            },
            max_len=args.seqlen,
            classes=args.code,
        )
        model_name = 'code'
    model = model.to(device).train()

    if args.continuous_decode:
        cmd_encoder = CMDEncoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        max_len=200,
        code_len = 4,
        num_code = 500,
        )
        cmd_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'cmdenc_epoch_300.pt')))
        cmd_encoder = cmd_encoder.to(device)

        param_encoder = PARAMEncoder(
            config={
                'hidden_dim': 512,
                'embed_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout_rate': 0.1
            },
            quantization_bits=args.bit,
            max_len=200,
            code_len = 2,
            num_code = 1000,
        )
        param_encoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'paramenc_epoch_300.pt')))
        param_encoder = param_encoder.to(device)
        
        ext_encoder = EXTEncoder(
            config={
                'hidden_dim': 512,
                'embed_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout_rate': 0.1
            },
            quantization_bits=args.bit,
            max_len=96,
            code_len = 4,
            num_code = 1000,
        )
        ext_encoder.load_state_dict(torch.load(os.path.join(args.ext_weight, 'extenc_epoch_200.pt')))
        ext_encoder = ext_encoder.to(device)

        cmd_codebook = cmd_encoder.vq_vae._embedding
        param_codebook = param_encoder.vq_vae._embedding
        ext_codebook = ext_encoder.vq_vae._embedding
    
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
            

            if args.continuous_decode:
                cmd_code = code[:,:4] 
                param_code = code[:,4:6] 
                ext_code = code[:,6:]
                cmd_embed = cmd_codebook(cmd_code)
                param_embed = param_codebook(param_code)
                ext_embed = ext_codebook(ext_code)
                embed = torch.cat([cmd_embed, param_embed, ext_embed], dim=1)
                arglist = [embed[:,:-1,:]]
                if args.use_voxels:
                    arglist.append(voxels)
                preds = model(*arglist)
                code_loss = F.mse_loss(preds, embed)
            else:
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
    parser.add_argument("--splits_file", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_encoder", action='store_false', dest='encoder')
    parser.add_argument("--continuous_decode", action='store_true')
    parser.add_argument("--sketch_weight", type=str)
    parser.add_argument("--ext_weight", type=str)
    parser.add_argument("--bit", type=int, default=6)


    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)
