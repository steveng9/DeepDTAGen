"""Train + evaluate a simplified affinity-prediction model (CNN or GNN).

Usage:
    python train.py --model cnn --dataset davis --epochs 100
    python train.py --model gnn --dataset davis --epochs 100

Both models predict affinity only (single MSE objective) -- none of
DeepDTAGen's generation machinery (VAE / Transformer decoder / FetterGrad).
Reports MSE / RMSE / CI / rm2 / Pearson / Spearman and per-epoch wall time.
"""
import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from metrics import all_metrics
from models import CNNDTA, GNNDTA, count_params

SEED = 4221


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(model_kind, dataset, batch_size):
    if model_kind == "cnn":
        from torch.utils.data import DataLoader
        from data import CNNDataset
        train_ds = CNNDataset(dataset, "train")
        test_ds = CNNDataset(dataset, "test")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        from torch_geometric.loader import DataLoader as GeoLoader
        from data import build_graph_dataset
        print("Building molecular graphs (in-the-clear encoding stage)...")
        train_list = build_graph_dataset(dataset, "train")
        test_list = build_graph_dataset(dataset, "test")
        train_loader = GeoLoader(train_list, batch_size=batch_size, shuffle=True)
        test_loader = GeoLoader(test_list, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_device(batch, model_kind, device):
    if model_kind == "cnn":
        xd, xt, y = batch
        return (xd.to(device), xt.to(device), None), y.to(device)
    batch = batch.to(device)
    return batch, batch.y.view(-1)


def evaluate(model, loader, model_kind, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inp, y = to_device(batch, model_kind, device)
            preds.append(model(inp).cpu().numpy())
            trues.append(y.cpu().numpy())
    P = np.concatenate(preds).flatten()
    G = np.concatenate(trues).flatten()
    return all_metrics(G, P), G, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "gnn"], required=True)
    ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool", choices=["max", "mean"], default="max",
                    help="global pooling; mean is MPC-friendly, max matches baseline")
    ap.add_argument("--head-dim", type=int, default=1024,
                    help="width of first FC head layer (1024=baseline, 512=reduced)")
    ap.add_argument("--head-layers", type=int, choices=[1, 2], default=2,
                    help="number of hidden FC layers in head (2=baseline, 1=drop 2nd)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed; vary across runs to average over inits")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the run tag so parallel runs don't collide")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--cuda", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None
                          and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{args.dataset}{args.tag_suffix}"
    print(f"=== {tag} | device={device} | pool={args.pool} | head_dim={args.head_dim} "
          f"| head_layers={args.head_layers} | seed={args.seed} ===")

    train_loader, test_loader = build_loaders(args.model, args.dataset, args.batch_size)

    if args.model == "cnn":
        model = CNNDTA(pool=args.pool, head_dim=args.head_dim,
                       head_layers=args.head_layers).to(device)
    else:
        model = GNNDTA(pool=args.pool, head_dim=args.head_dim,
                       head_layers=args.head_layers).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = {"MSE": float("inf")}
    history = []
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for batch in train_loader:
            inp, y = to_device(batch, args.model, device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        dt = time.time() - t0
        epoch_times.append(dt)
        train_mse = running / n

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m, G, P = evaluate(model, test_loader, args.model, device)
            history.append({"epoch": epoch, "train_mse": train_mse, **m})
            print(f"[{tag}] epoch {epoch:3d} | {dt:5.1f}s | train_mse {train_mse:.4f} "
                  f"| test MSE {m['MSE']:.4f} CI {m['CI']:.4f} rm2 {m['rm2']:.4f} "
                  f"Pearson {m['Pearson']:.4f}")
            if m["MSE"] < best["MSE"]:
                best = {"epoch": epoch, **m}
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"{tag}_best.pth"))
                np.savetxt(os.path.join(args.out_dir, f"{tag}_pred.txt"), P)
                np.savetxt(os.path.join(args.out_dir, f"{tag}_true.txt"), G)

    summary = {
        "tag": tag, "model": args.model, "dataset": args.dataset,
        "epochs": args.epochs, "params": count_params(model),
        "mean_epoch_sec": float(np.mean(epoch_times)),
        "total_train_sec": float(np.sum(epoch_times)),
        "best": best, "pool": args.pool, "head_dim": args.head_dim,
        "head_layers": args.head_layers, "seed": args.seed, "history": history,
    }
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BEST [{tag}] === epoch {best.get('epoch')} | "
          f"MSE {best['MSE']:.4f} | CI {best['CI']:.4f} | rm2 {best['rm2']:.4f}")
    print(f"params {summary['params']:,} | mean {summary['mean_epoch_sec']:.1f}s/epoch "
          f"| total {summary['total_train_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
