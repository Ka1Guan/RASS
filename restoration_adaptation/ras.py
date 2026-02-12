import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from restoration_adaptation.models.autoencoder_kl import AutoencoderKL
from restoration_adaptation.models.unet_2d_condition import UNet2DConditionModel
# from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
# from diffusers.models import UNet2DConditionModel
from peft import LoraConfig


def make_1step_sched(args):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(args.pretrained_sd_model_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step
    
class Generator_with_pretrain(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_sd_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_sd_model_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args)
        self.lora_rank_unet = args.lora_rank_unet

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_sd_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_sd_model_path, subfolder="unet")

        if args.pretrained_res_model_path is not None:
            print(f'Load Pretrained')
            sd = torch.load(args.pretrained_res_model_path)
            self.load_ckpt_from_state_dict(sd)

        # merge lora
        if args.merge_and_unload_lora:
            print(f'MERGE LORA')
            self.unet = self.unet.merge_and_unload()

        # build lora
        self.l_target_modules_encoder, self.l_target_modules_decoder, self.l_modules_others = [], [], []
        self.unet_lora(rank=self.lora_rank_unet)
        
        if args.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([1], device="cuda").long() # self.stage1_args.timesteps
        self.text_encoder.requires_grad_(False)

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    
    def unet_lora(self, rank):
        self.unet.requires_grad_(False)
        self.unet.train()

        l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
        for n, p in self.unet.named_parameters():
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if pattern in n and ("down_blocks" in n or "conv_in" in n):
                    self.l_target_modules_encoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                    self.l_target_modules_decoder.append(n.replace(".weight",""))
                    break
                elif pattern in n:
                    self.l_modules_others.append(n.replace(".weight",""))
                    break
        if 'conv_out' not in l_grep:
            if 'conv_out' in self.l_modules_others:
                self.l_modules_others.remove('conv_out')
            if 'conv_out' in self.l_target_modules_decoder:
                self.l_target_modules_decoder.remove('conv_out')
        lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=self.l_target_modules_encoder)
        lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=self.l_target_modules_decoder)
        lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=self.l_modules_others)
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder_s2")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder_s2")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others_s2")
        
    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae.requires_grad_(False)
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                caption = caption.strip("[] ,")
                caption = caption.replace("'", "")
                caption = " ".join(caption.split(","))
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None):

        encoded_control, out_feature_encoder = self.vae.encode(c_t)
        encoder_features = [out_feature_encoder[1], out_feature_encoder[3]]
        encoded_control = encoded_control.latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(batch["prompt"])
        pos_caption_enc = prompt_embeds

        _, unet_features = self.unet(encoded_control, self.timesteps, encoder_hidden_states=pos_caption_enc.to(torch.float32),)
        # model_pred = model_pred.sample
        # x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        # _, out_feature_decoder  = self.vae.decode(x_denoised / self.vae.config.scaling_factor)
        # decoder_features = [out_feature_decoder[0], out_feature_decoder[1]]

        out_features = encoder_features + unet_features

        return out_features
    

