"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Implements a feed-forward network with learnable gate parameters that
dynamically prune themselves during training via L1 sparsity regularization.

Author: AI Engineer Case Study Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 – PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that attaches a learnable gate scalar
    to every weight element.

    Forward pass:
        gates       = sigmoid(gate_scores)           ∈ (0, 1)
        pruned_w    = weight ⊙ gates                 element-wise product
        out         = x @ pruned_w.T + bias          standard affine map

    Because all operations are differentiable, autograd propagates gradients
    to *both* `weight` and `gate_scores` without any special handling.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias – initialised just like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Gate scores: one scalar per weight element, initialised near 1
        # so that at the start of training nearly all gates are open (≈ 0.73).
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – squash gate_scores into (0,1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2 – mask the weights
        pruned_weights = self.weight * gates             # element-wise ⊙

        # Step 3 – standard linear operation (gradients flow through both
        #           pruned_weights → weight  AND  pruned_weights → gate_scores)
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    def sparsity_info(self, threshold: float = 1e-2) -> Tuple[int, int, float]:
        """Returns (pruned_count, total_count, sparsity_fraction)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            total   = gates.numel()
            pruned  = (gates < threshold).sum().item()
        return int(pruned), total, pruned / total

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 (32×32×3 → 10 classes).
    Uses PrunableLinear throughout so every weight can be gated away.
    """

    def __init__(self):
        super().__init__()
        # CIFAR-10 images are 3×32×32 = 3072 raw pixels after flattening
        self.layers = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten: (B, 3072)
        return self.layers(x)

    # ------------------------------------------------------------------
    def prunable_layers(self) -> List[PrunableLinear]:
        """Returns all PrunableLinear modules in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    # ------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across all PrunableLinear layers.
        Minimising this sum drives gates toward 0 (pruning connections).
        """
        gate_values = [
            torch.sigmoid(layer.gate_scores)
            for layer in self.prunable_layers()
        ]
        return torch.cat([g.reshape(-1) for g in gate_values]).sum()

    # ------------------------------------------------------------------
    def report_sparsity(self, threshold: float = 1e-2) -> dict:
        """Aggregate sparsity report across all prunable layers."""
        total_pruned = total_weights = 0
        layer_reports = []
        for i, layer in enumerate(self.prunable_layers()):
            p, t, frac = layer.sparsity_info(threshold)
            total_pruned  += p
            total_weights += t
            layer_reports.append({
                "layer": i,
                "pruned": p,
                "total": t,
                "sparsity_pct": round(frac * 100, 2)
            })
        return {
            "layers": layer_reports,
            "overall_sparsity_pct": round(total_pruned / total_weights * 100, 2),
            "total_pruned": total_pruned,
            "total_weights": total_weights,
        }

    # ------------------------------------------------------------------
    def all_gate_values(self) -> torch.Tensor:
        """Flat tensor of all gate values (after sigmoid)."""
        with torch.no_grad():
            parts = [
                torch.sigmoid(layer.gate_scores).reshape(-1)
                for layer in self.prunable_layers()
            ]
        return torch.cat(parts).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 – Loss function  (defined inline in training loop, shown here for docs)
# ─────────────────────────────────────────────────────────────────────────────
#
#   total_loss = cross_entropy(logits, labels)  +  λ * sparsity_loss
#
#   cross_entropy  → encourages correct classification
#   λ * sparsity_loss → penalises non-zero gates, driving them to 0
#


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 – Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_data_loaders(batch_size: int = 128, data_root: str = "./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       SelfPruningNet,
    loader:      DataLoader,
    optimizer:   optim.Optimizer,
    lam:         float,
    device:      torch.device,
    epoch:       int,
    print_every: int = 200,
) -> Tuple[float, float]:
    """
    Trains for one epoch.
    Returns (avg_cls_loss, avg_sparsity_loss).
    """
    model.train()
    total_cls = total_spar = 0.0

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(images)

        # Part 2: Total loss = CE + λ * L1(gates)
        cls_loss  = F.cross_entropy(logits, labels)
        spar_loss = model.sparsity_loss()
        loss      = cls_loss + lam * spar_loss

        # Backward – gradients flow to weights AND gate_scores
        loss.backward()
        optimizer.step()

        total_cls  += cls_loss.item()
        total_spar += spar_loss.item()

        if (step + 1) % print_every == 0:
            print(f"  [Epoch {epoch:02d}] step {step+1}/{len(loader)} | "
                  f"CE={cls_loss.item():.4f}  Spar={spar_loss.item():.2f}  "
                  f"Total={loss.item():.4f}")

    n = len(loader)
    return total_cls / n, total_spar / n


def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Returns top-1 accuracy on the given data loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment for one λ value
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lam:          float,
    epochs:       int   = 30,
    lr:           float = 1e-3,
    batch_size:   int   = 128,
    device:       torch.device = torch.device("cpu"),
    data_root:    str   = "./data",
    seed:         int   = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  λ = {lam}   |   epochs = {epochs}   |   lr = {lr}")
    print(f"{'='*60}")

    train_loader, test_loader = get_data_loaders(batch_size, data_root)

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"cls_loss": [], "spar_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        cls_l, spar_l = train_one_epoch(
            model, train_loader, optimizer, lam, device, epoch)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        history["cls_loss"].append(cls_l)
        history["spar_loss"].append(spar_l)
        history["test_acc"].append(acc)

        spar_report = model.report_sparsity()
        print(f"  → Epoch {epoch:02d} | TestAcc={acc*100:.2f}%  "
              f"Sparsity={spar_report['overall_sparsity_pct']}%")

    # Final evaluation
    final_acc = evaluate(model, test_loader, device)
    spar_report = model.report_sparsity()
    gate_values = model.all_gate_values().numpy()

    print(f"\n  FINAL → Accuracy={final_acc*100:.2f}%  "
          f"Sparsity={spar_report['overall_sparsity_pct']}%")

    return {
        "lam":          lam,
        "final_acc":    final_acc,
        "sparsity_pct": spar_report["overall_sparsity_pct"],
        "spar_report":  spar_report,
        "gate_values":  gate_values,
        "history":      history,
        "model":        model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(gate_values: np.ndarray, lam: float, path: str):
    """
    Plots histogram of final gate values.
    A successful run shows a large spike near 0 and a cluster toward 1.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gate_values, bins=100, color="#2563eb", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Gate value  (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=13, fontweight="bold")
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
               label="Pruning threshold (0.01)")
    ax.legend()
    pct_pruned = (gate_values < 0.01).mean() * 100
    ax.text(0.65, 0.88, f"Pruned: {pct_pruned:.1f}%",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor="lightyellow", edgecolor="gray", boxstyle="round"))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved gate distribution plot → {path}")


def plot_training_curves(results_list: list, path: str):
    """Plots test accuracy over epochs for all λ values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for res, color in zip(results_list, colors):
        lam = res["lam"]
        acc_pct = [a * 100 for a in res["history"]["test_acc"]]
        axes[0].plot(acc_pct, label=f"λ={lam}", color=color, linewidth=1.8)
        axes[1].plot(res["history"]["spar_loss"], label=f"λ={lam}",
                     color=color, linewidth=1.8)

    axes[0].set_title("Test Accuracy vs Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Sparsity Loss vs Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("L1 gate sum")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Self-Pruning Network – Training Dynamics", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {path}")


def plot_combined_gate_distributions(results_list: list, path: str):
    """Side-by-side gate histograms for all λ values."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for ax, res, color in zip(axes, results_list, colors):
        gv   = res["gate_values"]
        lam  = res["lam"]
        pct  = (gv < 0.01).mean() * 100
        ax.hist(gv, bins=100, color=color, edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"λ = {lam}\n(Sparsity: {pct:.1f}%)", fontweight="bold")
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Count")

    plt.suptitle("Gate Value Distributions for Different λ Values", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved combined gate distributions → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_ROOT  = "./data"
    OUT_DIR    = "./outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Three λ values: low / medium / high ──────────────────────────────────
    LAMBDA_VALUES = [1e-5, 1e-4, 5e-4]
    EPOCHS        = 30        # increase to 60+ for production runs
    LR            = 1e-3
    BATCH_SIZE    = 128

    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(
            lam       = lam,
            epochs    = EPOCHS,
            lr        = LR,
            batch_size= BATCH_SIZE,
            device    = device,
            data_root = DATA_ROOT,
        )
        all_results.append(result)

        # Per-λ gate distribution plot
        plot_gate_distribution(
            result["gate_values"],
            lam,
            os.path.join(OUT_DIR, f"gate_dist_lambda_{lam}.png")
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'λ':>10}  {'Test Acc (%)':>14}  {'Sparsity (%)':>14}")
    print("-"*55)
    for r in all_results:
        print(f"{r['lam']:>10}  {r['final_acc']*100:>14.2f}  "
              f"{r['sparsity_pct']:>14.2f}")
    print("="*55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(
        all_results,
        os.path.join(OUT_DIR, "training_curves.png")
    )
    plot_combined_gate_distributions(
        all_results,
        os.path.join(OUT_DIR, "gate_distributions_all.png")
    )

    # Save numeric summary as JSON (handy for report generation)
    summary = [
        {
            "lambda": r["lam"],
            "test_acc_pct": round(r["final_acc"]*100, 2),
            "sparsity_pct": r["sparsity_pct"],
        }
        for r in all_results
    ]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {os.path.join(OUT_DIR, 'summary.json')}")
    print("Done.")
