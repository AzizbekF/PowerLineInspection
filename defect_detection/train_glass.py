import argparse, os, time, warnings
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from utils.utils import set_seed, train_one_epoch, evaluate, calc_pos_weight
from dataset.DefectDataset import DefectDataset

warnings.filterwarnings("ignore")

def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data augmentation + preprocessing
    train_tfms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # split 90 / 10
    full_df = pd.read_csv(args.csv)
    val_df = full_df.sample(frac=0.1, random_state=42)
    train_df = full_df.drop(val_df.index)
    train_df.to_csv("train_tmp.csv", index=False)
    val_df.to_csv("val_tmp.csv", index=False)

    train_ds = DefectDataset("train_tmp.csv", args.root, train_tfms)
    val_ds   = DefectDataset("val_tmp.csv",   args.root, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,num_workers=4, pin_memory=True)

    # class imbalance handling
    pos_weight = calc_pos_weight(train_ds.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # model: pretrained ResNet‑50 as a feature extractor
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_path = 0.0, "defect_resnet50.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train loss {tr_loss:.4f}  acc {tr_acc*100:.1f}% | "
              f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}%  "
              f"({time.time()-t0:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"\n✓ Training complete. Best val acc: {best_acc*100:.2f}%  → saved to {best_path}")

    # quick test on a single image
    if args.test_img:
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        img = val_tfms(Image.open(args.test_img).convert("RGB")).unsqueeze(0).to(device)
        pred = model(img).sigmoid().item()
        print(f"\n{os.path.basename(args.test_img)}  defect probability: {pred:.3f}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = pd.read_csv('image_labels_with_code.csv')
    df = df[df["category_code"] == 0]

    p = argparse.ArgumentParser(description="Binary defect classifier")
    p.add_argument("--csv",   required=True, help="CSV file with image_path + status")
    p.add_argument("--root",  required=True, help="Root folder containing images")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs",     type=int, default=16, help="batch size")
    p.add_argument("--test_img", default=None, help="Optional: run a single‑image test at the end")
    args = p.parse_args()
    main(args)