import torch
import numpy as np
from tdqm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATNENT_WIDTH = WIDTH // 8
LATNENT_HEIGHT = HEIGHT // 8

def generate(
    prompt: str, 
    uncond_prompt: str, 
    input_image=None, 
    strength=0.8, 
    do_cfg=True, 
    cfg_scale=7.5, 
    sampler_name='ddqm',
    n_inference=50,
    model={}, 
    seed=None,
    device=None,  
    idle_device=None,
    tokenizer=None
):

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1]")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
    
        generator = torch.Generator(device=device)
            
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        # with cfg, the model is inferenced twice, once with the prompt and once with the uncond_prompt
        if do_cfg:
            # convert prompts to tokenized format with max_length 77 based on clip tokenzier
            # take input_ids of the tokenized prompts
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            # (batch_size, seq_len) -> (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, device=device, dtype=torch.long)

            # (batch_size, seq_len) -> (batch_size, seq_len, dim=768)
            cond_context = clip(cond_tokens)



            prompts = [uncond_prompt, prompt]
        else:
            prompts = [prompt]