from .layers.transformer import *
from .layers.improved_transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 


SAMPLE_PROB = 0.99 


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class CodeModel(nn.Module):

  def __init__(self,
               config,
               max_len=8,
               classes = 512,
               name='ar_model'):
    super(CodeModel, self).__init__()
    self.embed_dim = config['embed_dim']
    self.max_len = max_len
    self.dropout = config['dropout_rate']

    # Position embeddings
    self.pos_embed = PositionalEncoding(max_len=self.max_len, d_model=self.embed_dim)
   
    # Discrete vertex value embeddings
    self.embed = Embedder(classes, self.embed_dim)
  
    # Transformer decoder
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'],
                        nhead=config['num_heads'], dropout=self.dropout)
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)
    self.fc = nn.Linear(self.embed_dim, classes)
    

  def forward(self, code):
    """ forward pass """
    if code[0] is None:
      bs = len(code)
      seq_len = 0
    else:
      bs, seq_len = code.shape[0], code.shape[1]

    # Context embedding values
    context_embedding = torch.zeros((bs, 1, self.embed_dim)).cuda() # [bs, 1, dim]
      
    if seq_len > 0:
      embeddings = self.embed(code.flatten()).view(bs, code.shape[1], self.embed_dim)  # [bs, seqlen, dim]
      decoder_inputs = torch.cat([context_embedding, embeddings], axis=1) # [bs, seqlen+1, dim]
      # Positional embedding
      decoder_inputs = self.pos_embed(decoder_inputs.transpose(0,1))   # [seqlen+1, bs, dim]
    
    else:
      decoder_inputs = self.pos_embed(context_embedding.transpose(0,1))   # [1, bs, dim]
      
    memory = torch.zeros((1, bs, self.embed_dim)).cuda()
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.shape[0]).cuda()  # masked with -inf
    decoder_outputs = self.decoder(tgt=decoder_inputs, memory=memory, memory_key_padding_mask=None,
                                   tgt_mask=nopeak_mask, tgt_key_padding_mask=None)
    
    # Get logits 
    logits = self.fc(decoder_outputs)
    return logits.transpose(0,1)
    

  def sample(self, n_samples=10):
    """
    sample from distribution (top-k, top-p)
    """
    #samples = []
    temperature = 1.0
    top_k = 0
    top_p = SAMPLE_PROB
    v_seq = [None] * n_samples
    for k in range(self.max_len):          
         
        # pass through decoder
        with torch.no_grad():
          logits = self.forward(code=v_seq)
          logits = logits[:, -1, :] / temperature
        
        # Top-p sampling 
        next_vs = []
        for logit in logits:   
            filtered_logits = top_k_top_p_filtering(logit.clone(), top_k=top_k, top_p=top_p)
            next_v = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
            next_vs.append(next_v.item())

        # Add next tokens
        next_seq = torch.LongTensor(next_vs).view(len(next_vs), 1).cuda()
        if v_seq[0] is None:
            v_seq = next_seq
        else:
            v_seq = torch.cat([v_seq, next_seq], 1)
       
    return v_seq



class CondARModel(nn.Module):
  """Autoregressive generative model of quantized mesh vertices."""

  def __init__(self,
               config,
               encoder_config,
               max_len=8,
               classes = 512,
               name='ar_model',
               use_transformer_encoder=True):
    super(CondARModel, self).__init__()

    self.embed_dim = config['embed_dim']
    self.max_len = max_len
    self.dropout = config['dropout_rate']
    self.use_transformer_encoder = use_transformer_encoder

    # Position embeddings
    self.pos_embed = PositionalEncoding(max_len=self.max_len, d_model=self.embed_dim)
   
    # Discrete vertex value embeddings
    self.code_embed = Embedder(classes, self.embed_dim)
    #self.cond_embed = Embedder(classes, self.embed_dim)
    self.pos_embed_cond = PositionalEncoding(max_len=512, d_model=self.embed_dim)
  
    # Transformer decoder
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'],
                        nhead=config['num_heads'], dropout=self.dropout)
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)
    self.fc = nn.Linear(self.embed_dim, classes)
    self.encoder = VoxelEncoder(config['embed_dim'], encoder_config['hidden_dim'], encoder_config['num_heads'], config['dropout_rate'], encoder_config['num_layers'], use_transformer_encoder=use_transformer_encoder)
    

  def forward(self, code, cond):
    """ forward pass """
    if code[0] is None:
      bs = len(code)
      seq_len = 0
    else:
      bs, seq_len = code.shape[0], code.shape[1]

    # Context embedding
    context_embedding = torch.zeros((bs, 1, self.embed_dim)).cuda() # [bs, 1, dim]
    
    # Code seq embedding 
    if seq_len > 0:
      embeddings = self.code_embed(code.flatten()).view(bs, code.shape[1], self.embed_dim)  # [bs, seqlen, dim]
      decoder_inputs = torch.cat([context_embedding, embeddings], axis=1) # [bs, seqlen+1, dim]
      # Positional embedding
      decoder_inputs = self.pos_embed(decoder_inputs.transpose(0,1))   # [seqlen+1, bs, dim]
    else:
      decoder_inputs = self.pos_embed(context_embedding.transpose(0,1))   # [1, bs, dim]

    # Cond input embedding 
    #cond_input = self.cond_embed(cond.flatten()).view(bs, cond.shape[1], self.embed_dim) # [bs, seqlen, dim]
    cond_encoded = self.encoder(cond) #[bs, res, res, res, dim]
    if self.use_transformer_encoder:
      memory = cond_encoded#.transpose(0,1)
    else:
      cond_encoded = cond_encoded.view(bs, -1, self.embed_dim)
      memory = self.pos_embed_cond(cond_encoded.transpose(0, 1))
    # Pass through AR decoder
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.shape[0]).cuda()  # masked with -inf
    decoder_outputs = self.decoder(tgt=decoder_inputs, memory=memory, tgt_mask=nopeak_mask)
   
    # Get logits 
    logits = self.fc(decoder_outputs)
    return logits.transpose(0,1)


  def sample(self, n_samples, cond_code, deterministic=False):
    """
    sample from distribution (top-k, top-p)
    """
    temperature = 1.0
    top_k = 0
    top_p = SAMPLE_PROB
    v_seq = [None] * n_samples
    for k in range(self.max_len):          
         
        # pass through decoder
        with torch.no_grad():
          logits = self.forward(code=v_seq, cond=cond_code)
          logits = logits[:, -1, :] / temperature
        
        # Top-p sampling 
        if deterministic:
          next_seq = torch.argmax(logits, 1).unsqueeze(1)
        else:
          next_vs = []
          for logit in logits:   
              filtered_logits = top_k_top_p_filtering(logit.clone(), top_k=top_k, top_p=top_p)
              next_v = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
              next_vs.append(next_v.item())
          # Add next tokens
          next_seq = torch.LongTensor(next_vs).view(len(next_vs), 1).cuda()
        if v_seq[0] is None:
            v_seq = next_seq
        else:
            v_seq = torch.cat([v_seq, next_seq], 1)
       
    return v_seq


class VoxelEncoder(nn.Module):
  def __init__(self, model_dim = 128, hidden_dim=512, num_heads=8, dropout_rate=0.0, num_transformer_layers=4, use_transformer_encoder=True):
    super(VoxelEncoder, self).__init__()
    self.embed_dim = model_dim
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.dropout = dropout_rate
    self.num_layers = num_transformer_layers
    self.use_transformer_encoder = use_transformer_encoder

    self.conv_1 = nn.Conv3d(1, 32, 3, padding=1)  # out: 32
    self.conv_1_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
    self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
    self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
    self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
    self.conv_3_1 = nn.Conv3d(128, self.embed_dim, 3, padding=1)  # out: 8

    self.actvn = nn.ReLU()

    self.maxpool = nn.MaxPool3d(2)

    self.conv1_1_bn = nn.BatchNorm3d(64)
    self.conv2_1_bn = nn.BatchNorm3d(128)
    self.conv3_1_bn = nn.BatchNorm3d(model_dim)

    self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=8**3)

    if use_transformer_encoder:
      encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=self.num_heads, 
                                              dim_feedforward=self.hidden_dim, dropout=self.dropout)
      encoder_norm = LayerNorm(self.embed_dim)
      self.encoder = TransformerEncoder(encoder_layers, self.num_layers, encoder_norm)
  
  def forward(self, x):
    net = self.actvn(self.conv_1(x))
    net = self.actvn(self.conv_1_1(net))
    net = self.conv1_1_bn(net)
    net = self.maxpool(net)

    net = self.actvn(self.conv_2(net))
    net = self.actvn(self.conv_2_1(net))
    net = self.conv2_1_bn(net)
    net = self.maxpool(net)

    net = self.actvn(self.conv_3(net))
    net = self.actvn(self.conv_3_1(net))
    net = self.conv3_1_bn(net) #embed_dimx8x8x8

    if self.use_transformer_encoder:
      encoder_embed = net.view(net.shape[0], net.shape[1], -1)
      encoder_embed = torch.permute(encoder_embed, (2, 0, 1))
      encoder_input = self.pos_embed(encoder_embed)
      outputs = self.encoder(src=encoder_input)

      return outputs
    else:
      return net
