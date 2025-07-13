# backend/train.py

# backend/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from model.student_model import StudentNet
from model.teacher_model import TeacherUNet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_images = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(".png")])
        self.hr_images = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = cv2.imread(self.lr_images[idx])
        hr = cv2.imread(self.hr_images[idx])

        lr = cv2.resize(lr, (256, 256))
        hr = cv2.resize(hr, (256, 256))

        lr_tensor = torch.FloatTensor(lr / 255.0).permute(2, 0, 1)
        hr_tensor = torch.FloatTensor(hr / 255.0).permute(2, 0, 1)

        return lr_tensor, hr_tensor

def train(epochs=5, batch_size=4, learning_rate=1e-4):
    dataset = DIV2KDataset("../dataset/lr", "../dataset/hr")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student = StudentNet().to(device)
    teacher = TeacherUNet().to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loss_fn_l1 = nn.L1Loss()
    loss_fn_distill = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        student.train()

        for lr_img, hr_img in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            with torch.no_grad():
                teacher_out = teacher(lr_img)

            student_out = student(lr_img)
            loss_recon = loss_fn_l1(student_out, hr_img)
            loss_distill = loss_fn_distill(student_out, teacher_out)
            loss = loss_recon + 0.5 * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    torch.save(student.state_dict(), "student_weights.pth")
    print("✅ Training complete. Model saved as student_weights.pth")

if __name__ == "__main__":
    train()
