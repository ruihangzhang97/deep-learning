import torch
import numpy as np
from tdqm import tqdm
from sd.ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(
    prompt: str, 
    uncond_prompt: str, 
    input_image=None, 
    strength=0.8,
    do_cfg=True, 
    cfg_scale=7.5, 
    sampler_name='ddqm',
    n_inference_steps=50,
    models={}, 
    seed=None,
    device=None,  
    idle_device=None,
    tokenizer=None
):

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1)")
        
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

            # convert uncond_prompt to tokenized format with max_length 77 based on clip tokenizer
            # take input_ids of the tokenized uncond_prompt
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len) -> (batch_size, seq_len)
            uncond_tokens = torch.tensor(uncond_tokens, device=device, dtype=torch.long)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim=768)
            uncond_context = clip(uncond_tokens)

            # concatenate the contexts -> (2, seq_len=77, dim=768)
            context = torch.cat([cond_context, uncond_context], dim=0)

        else:
            # convert into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, device=device, dtype=torch.long)
            
            # (batch_size=1. seq_len=77) -> (batch_size=1, seq_len=77, dim=768)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == 'ddqm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} is not supported")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (height, width, channels) -> (batch_size, height, width, channels) -> (batch_size=1, channels=3, height=512, width=512)
            # to match the input shape of the encoder
            input_image_tensor =  input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder.encode(input_image_tensor, noise=encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0]) # adding noise to image latents based on the strength value

            to_idle(encoder)

        else:
            # start with random noise if we are doing text to image generation
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tdqm(sampler.timesteps)

        # denoise image for each timestep
        for i, timestep in enumerate(timesteps):

            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latent_height, latent_width): input to encoder
            model_input = latents
            if do_cfg:
                # (batch_size, 4, latent_height, latent_width) -> (batch_size=2, 4, latent_height, latent_width)
                # repeat the model input for the conditional and unconditional contexts
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # (batch_size=2, 4, latent_height, latent_width) -> (batch_size=1, 4, latent_height, latent_width)
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # remove noise predicted by the unet
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (batch_size=1, channels=4, height=64, width=64) -> (batch_size=1, height=64, width=64, channels=4)
        iamges = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0] 

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

