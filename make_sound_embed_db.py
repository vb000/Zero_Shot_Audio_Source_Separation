import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa

from models.htsat import HTSAT_Swin_Transformer
import htsat_config

class HTSAT(nn.Module):
    def __init__(self):
        super(HTSAT, self).__init__()
        self.freq_ratio = htsat_config.htsat_spec_size // htsat_config.mel_bins
        self.target_T = int(htsat_config.hop_size * htsat_config.htsat_spec_size * self.freq_ratio) - 1
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=htsat_config.htsat_spec_size,
            patch_size=htsat_config.htsat_patch_size,
            in_chans=1,
            num_classes=htsat_config.classes_num,
            window_size=htsat_config.htsat_window_size,
            config=htsat_config,
            depths=htsat_config.htsat_depth,
            embed_dim=htsat_config.htsat_dim,
            patch_stride=htsat_config.htsat_stride,
            num_heads=htsat_config.htsat_num_head
        )
    
    def forward(self, x, infer_mode=False):
        return self.sed_model(x, infer_mode)

    @torch.no_grad()
    def embed_sound(self, f, device):
        audio, sr = librosa.load(f, sr=None)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        audio = audio[:, :self.target_T].to(device)
        return self.forward(audio)['latent_output']

def make_model(ckpt_path):
    model = HTSAT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    return model

def make_sound_embed_db(model, src, dest, device):
    """
    Args:
        model: Embedding model implementing embed_sound method that takes audio path
            and returns embedding tensor of shape (1, embed_dim).
        src: Path to folder with audio files organized in subfolders by class label.
        tgt: Path to output directory for embedding database. Embeddings are stored in
            separate .pt for each class label. Each .pt file contains a (N + 1, embed_dim)
            tensor where N is the number of audio files in the class folder. The first
            row of the tensor contains the mean embedding of all audio files in the class.
    Returns:
        None
    """
    # Make output directory
    os.makedirs(dest, exist_ok=False)
    for label in os.listdir(src):
        embeds = []
        print(f"Processing class {label}")
        for f in tqdm(os.listdir(os.path.join(src, label))):
            embeds.append(model.embed_sound(os.path.join(src, label, f), device))
        embeds = torch.cat(embeds, dim=0)
        embed_mean = embeds.mean(dim=0, keepdim=True)
        embeds = torch.cat([embed_mean, embeds], dim=0).to("cpu")
        torch.save(embeds, os.path.join(dest, label + ".pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    model = make_model(args.ckpt_path).to(args.device)
    make_sound_embed_db(model, args.src, args.dest, args.device)
