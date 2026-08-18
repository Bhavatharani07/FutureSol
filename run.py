import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np

# Positional Command Line Arguments
if len(sys.argv) < 3:
    print("Usage: python run.py <input-dir> <output-dir>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
weights_path = os.path.join(os.path.dirname(__file__), "models", "best_attention_unet.pth")

os.makedirs(output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_l = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, l):
        g1, l1 = self.W_g(g), self.W_l(l)
        return l * self.psi(self.relu(g1 + l1))

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.pool1 = DoubleConv(1, 64), nn.MaxPool2d(2)
        self.conv2, self.pool2 = DoubleConv(64, 128), nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2, self.conv_up2 = AttentionGate(128, 128, 64), DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1, self.conv_up1 = AttentionGate(64, 64, 32), DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        b = self.bottleneck(self.pool2(c2))
        u2 = self.up2(b)
        c_up2 = self.conv_up2(torch.cat([u2, self.att2(u2, c2)], dim=1))
        u1 = self.up1(c_up2)
        c_up1 = self.conv_up1(torch.cat([u1, self.att1(u1, c1)], dim=1))
        return torch.clamp(self.final_conv(c_up1), 0.0, 1.0)

model = AttentionUNet().to(device)
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
model.eval()

input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

with torch.no_grad():
    for fpath in input_files:
        fname = os.path.basename(fpath)
        img = np.load(fpath).astype(np.float32)
        
        while img.ndim > 2: img = img[0]
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
            
        inp_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(inp_tensor).squeeze().cpu().numpy()
        
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred = np.clip(pred, 0.0, 1.0)
        
        save_path = os.path.join(output_dir, fname)
        np.save(save_path, pred)