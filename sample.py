import os
import torch
import argparse
from multiprocessing import Pool
from model.code import CodeModel, CondARModel
from model.decoder import SketchDecoder, EXTDecoder
from model.encoder import PARAMEncoder, CMDEncoder, EXTEncoder
from dataset import CodeDataset
import torch.utils.data

import sys
sys.path.insert(0, 'utils')
from utils import CADparser, write_obj_sample

NUM_TRHEADS = 36 
NUM_SAMPLE = 5000

def sample(args):
    # Initialize gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")
  
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
    cmd_encoder = cmd_encoder.to(device).eval()

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
    param_encoder = param_encoder.to(device).eval()

    sketch_decoder = SketchDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        pix_len=200,
        cmd_len=124,
        quantization_bits=args.bit,
    )
    sketch_decoder.load_state_dict(torch.load(os.path.join(args.sketch_weight, 'sketchdec_epoch_300.pt')))
    sketch_decoder = sketch_decoder.to(device).eval()
    
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
    ext_encoder = ext_encoder.to(device).eval()

    ext_decoder = EXTDecoder(
        config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 4, 
            'num_heads': 8,
            'dropout_rate': 0.1  
        },
        max_len=96,
        quantization_bits=args.bit,
    )
    ext_decoder.load_state_dict(torch.load(os.path.join(args.ext_weight, 'extdec_epoch_200.pt')))
    ext_decoder = ext_decoder.to(device).eval()

    if args.voxel_path is not None:
        dataset = CodeDataset(args.code_path, voxel_path=args.voxel_path, names_path=args.names_path, cache=True, splits_file=args.splits_file, mode=args.mode, res=args.res)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=args.batchsize, num_workers=5)
        code_model = CondARModel(config={
            'hidden_dim': 512,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.1,
        },
        max_len=10,
        classes=128 if args.continuous_decode else 1000,
        encoder_config={
            'hidden_dim': 256,
            'embed_dim': 256, 
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.1
        },
        use_transformer_encoder=args.encoder,
        use_transformer_decoder=args.transformer_decoder,
        continuous_decode=args.continuous_decode,
        )
    else:
        code_model = CodeModel(
            config={
                'hidden_dim': 512,
                'embed_dim': 256, 
                'num_layers': 8,
                'num_heads': 8,
                'dropout_rate': 0.1
            },
            max_len=10,
            classes=1000,
        )
    if args.code_weight is not None:
        code_model.load_state_dict(torch.load(args.code_weight))
    code_model = code_model.to(device).eval()

    print('Random Generation...') if args.voxel_path is None else print('Conditional Generation...')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    cad = []
    names = []
    cmd_codebook = cmd_encoder.vq_vae._embedding
    param_codebook = param_encoder.vq_vae._embedding
    ext_codebook = ext_encoder.vq_vae._embedding
    
    if args.voxel_path is not None:
        for batch in dataloader:
            if len(cad) >= NUM_SAMPLE:
                break
            voxels = batch['voxels'].to(device)
            name = batch['name']
            sample_merges, name = compute_sample(cmd_encoder, param_encoder, sketch_decoder, ext_encoder, ext_decoder, code_model, cmd_codebook, param_codebook, ext_codebook, conditioning=voxels, names=name)
            cad += sample_merges
            names += name
            print(f'cad:{len(cad)}/{len(dataset)}')
    else:
        while len(cad) < NUM_SAMPLE:
            sample_merges = compute_sample(cmd_encoder, param_encoder, sketch_decoder, ext_encoder, ext_decoder, code_model, cmd_codebook, param_codebook, ext_codebook)
            cad += sample_merges
            print(f'cad:{len(cad)}')
        
    # # Parallel raster OBJ
    gen_data = []

    load_iter = Pool(NUM_TRHEADS).imap(raster_cad, cad) 
    for data_sample in load_iter:
        gen_data += data_sample
    print(len(gen_data))

    print('Saving...')
    print('Writting OBJ...')
    for index, value in enumerate(gen_data):
        if args.voxel_path is None:
            output = os.path.join(args.output, str(index).zfill(5))
        else:
            output = os.path.join(args.output, names[index])

        if not os.path.exists(output):
            os.makedirs(output)
        write_obj_sample(output, value)

def compute_sample(cmd_encoder, param_encoder, sketch_decoder, ext_encoder, ext_decoder, code_model, cmd_codebook, param_codebook, ext_codebook, conditioning=None, names=None):
    with torch.no_grad():
        if conditioning is None:
            codes = code_model.sample(n_samples=args.batchsize)
        else:
            codes = code_model.sample(n_samples=args.batchsize, cond_code=conditioning, deterministic=args.deterministic)
        
        if args.continuous_decode:
            latent_cmd = codes[:,:4,:]
            latent_param = codes[:,4:6,:]
            latent_ext = codes[:,6:,:]

            latent_cmd = cmd_encoder.up(cmd_encoder.vq_vae(latent_cmd)[1])
            latent_param = param_encoder.up(param_encoder.vq_vae(latent_param)[1])
            latent_ext = ext_encoder.up(ext_encoder.vq_vae(latent_ext)[1])
            latent_sketch = torch.cat((latent_cmd, latent_param), 1)
            names_final = names
        else:
            cmd_code = codes[:,:4] 
            param_code = codes[:,4:6] 
            ext_code = codes[:,6:] 

            cmd_codes = []
            param_codes = []
            ext_codes = []
            names_final = []
            records = zip(cmd_code, param_code, ext_code) if names is None else zip(cmd_code, param_code, ext_code, names)
            #filter out invalid codes
            for record in records:
                if names is None:
                    cmd, param, ext = record
                else:
                    cmd, param, ext, name = record
                if torch.max(cmd) >= 500:
                    continue
                else:
                    cmd_codes.append(cmd)
                    param_codes.append(param)
                    ext_codes.append(ext)
                    if names is not None:
                        names_final.append(name)
            cmd_codes = torch.vstack(cmd_codes)
            param_codes = torch.vstack(param_codes)
            ext_codes = torch.vstack(ext_codes)

            latent_cmd = cmd_encoder.up(cmd_codebook(cmd_codes))
            latent_param = param_encoder.up(param_codebook(param_codes))
            latent_ext = ext_encoder.up(ext_codebook(ext_codes))
            latent_sketch = torch.cat((latent_cmd, latent_param), 1)
                
        # Parallel Sample Sketches 
    sample_pixels, latent_ext_samples = sketch_decoder.sample(n_samples=latent_sketch.shape[0], \
                        latent_z=latent_sketch, latent_ext=latent_ext)
    _latent_ext_ = torch.vstack(latent_ext_samples)

        # Parallel Sample Extrudes 
    sample_merges = ext_decoder.sample(n_samples=len(sample_pixels), latent_z=_latent_ext_, sample_pixels=sample_pixels)
    if names is None:
        return sample_merges
    else:
        return sample_merges, names_final


def raster_cad(pixels):   
    try:
        parser = CADparser(args.bit)
        parsed_data = parser.perform(pixels)
        return [parsed_data]
    except Exception as error_msg:  
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sketch_weight", type=str, required=True)
    parser.add_argument("--ext_weight", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bit", type=int, required=True)
    parser.add_argument("--voxel_path", type=str, default=None)
    parser.add_argument("--code_weight", type=str, default=None)
    parser.add_argument("--code_path", type=str, default=None)
    parser.add_argument("--names_path", type=str)
    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--no_cache", action='store_false', dest='cache')
    parser.add_argument("--splits_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--no_encoder", action='store_false', dest='encoder')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--continuous_decode", action='store_true')
    parser.add_argument("--no_transformer_decoder", action='store_false', dest='transformer_decoder')

    args = parser.parse_args()
    
    sample(args)

