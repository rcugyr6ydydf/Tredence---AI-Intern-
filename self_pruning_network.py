"""
Self-Pruning Neural Network on CIFAR-10
========================================
Author  : <Your Name>
Task    : Tredence AI Engineering Intern – Case Study
Description:
    Implements a feed-forward network whose weights are gated by learnable
    scalar parameters trained with an L1 sparsity penalty, enabling the
    network to prune itself during training.
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1. PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates a learnable
    scalar gate with every weight element.

    Forward pass:
        gates        = sigmoid(gate_scores)          # ∈ (0, 1)
        pruned_weight = weight ⊙ gates               # element-wise product
        output        = input @ pruned_weight.T + bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate scores – one per weight, initialised near 1 so training starts
        # with almost fully-active connections and prunes from there.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform for the weight (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Start gate_scores at 2.0 → sigmoid(2) ≈ 0.88, so most gates begin open
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)          # (out, in)
        pruned_weight = self.weight * gates                       # element-wise
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gates – encourages gates to collapse to 0."""
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────
# 2. Network Architecture
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A simple feed-forward network for CIFAR-10 (3×32×32 → 10 classes).
    All linear projections are PrunableLinear layers.
    """

    def __init__(self):
        super().__init__()
        # Small CNN front-end to extract spatial features cheaply
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 16×16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 8×8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),   # 4×4 → 256*4*4 = 4096
        )

        # Fully-connected head – all PrunableLinear
        self.classifier = nn.Sequential(
            PrunableLinear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 gate loss over all PrunableLinear layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total = total + m.sparsity_loss()
        return total

    def get_all_gates(self) -> np.ndarray:
        """Collect all gate values from every PrunableLinear layer."""
        gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates.append(m.get_gates().cpu().numpy().ravel())
        return np.concatenate(gates) if gates else np.array([])

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of weights whose gate < threshold."""
        gates = self.get_all_gates()
        if len(gates) == 0:
            return 0.0
        return float((gates < threshold).sum() / len(gates) * 100)


# ─────────────────────────────────────────────
# 3. Data Loading
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam: float, device):
    model.train()
    total_loss = cls_loss_sum = sp_loss_sum = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        cls_loss_sum += cls_loss.item()
        sp_loss_sum  += sp_loss.item()
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return {
        "total_loss": total_loss / n,
        "cls_loss":   cls_loss_sum / n,
        "sp_loss":    sp_loss_sum  / n,
        "train_acc":  correct / total * 100,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


# ─────────────────────────────────────────────
# 5. Full Experiment for One Lambda
# ─────────────────────────────────────────────

def run_experiment(lam: float, train_loader, test_loader, device,
                   epochs: int = 30, lr: float = 1e-3):
    print(f"\n{'='*55}")
    print(f"  Running experiment  λ = {lam}")
    print(f"{'='*55}")

    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        stats    = train_one_epoch(model, train_loader, optimizer, lam, device)
        test_acc = evaluate(model, test_loader, device)
        sparsity = model.compute_sparsity()
        scheduler.step()

        history.append({**stats, "test_acc": test_acc, "sparsity": sparsity})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"CLS {stats['cls_loss']:.4f} | "
                  f"SP {stats['sp_loss']:.2f} | "
                  f"Train {stats['train_acc']:.1f}% | "
                  f"Test {test_acc:.1f}% | "
                  f"Sparsity {sparsity:.1f}%")

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.compute_sparsity()
    final_gates    = model.get_all_gates()
    print(f"\n  ✓ Final → Test Acc: {final_test_acc:.2f}%  |  "
          f"Sparsity: {final_sparsity:.2f}%")

    return {
        "lam":        lam,
        "test_acc":   final_test_acc,
        "sparsity":   final_sparsity,
        "gates":      final_gates,
        "history":    history,
        "model":      model,
    }


# ─────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────

def plot_gate_distribution(results: list, save_path: str = "gate_distribution.png"):
    """
    Plot the gate value distribution for each lambda value.
    A successful result shows a spike at 0 and a separate cluster near 1.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    colors = ["#E07B54", "#5B9BD5", "#70AD47"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gates"]
        ax.hist(gates, bins=100, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.01)")
        ax.set_title(
            f"λ = {res['lam']}\n"
            f"Acc: {res['test_acc']:.1f}%  |  Sparsity: {res['sparsity']:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Gate Value Distribution After Training\n"
                 "(Spike at 0 → successful pruning)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot saved → {save_path}]")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Plot test accuracy and sparsity over epochs for all lambda values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E07B54", "#5B9BD5", "#70AD47"]
    styles = ["-", "--", "-."]

    for res, color, style in zip(results, colors, styles):
        epochs  = range(1, len(res["history"]) + 1)
        acc     = [h["test_acc"] for h in res["history"]]
        spar    = [h["sparsity"] for h in res["history"]]
        label   = f"λ={res['lam']}"
        ax1.plot(epochs, acc,  color=color, linestyle=style, lw=2, label=label)
        ax2.plot(epochs, spar, color=color, linestyle=style, lw=2, label=label)

    ax1.set_title("Test Accuracy vs Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.legend(); ax1.grid(linestyle=":", alpha=0.5)

    ax2.set_title("Sparsity Level vs Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Sparsity (%)")
    ax2.legend(); ax2.grid(linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved → {save_path}]")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Three lambda values: low / medium / high sparsity pressure
    lambdas = [1e-5, 1e-4, 5e-4]
    epochs  = 30           # increase to 50–60 for best accuracy on real hardware

    results = []
    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=epochs)
        results.append(res)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_gate_distribution(results, "gate_distribution.png")
    plot_training_curves(results,   "training_curves.png")

    # ── Console Summary Table ──────────────────────────────────────────────
    print("\n" + "─" * 52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print("─" * 52)
    for res in results:
        print(f"  {res['lam']:<12} {res['test_acc']:>14.2f}% {res['sparsity']:>15.2f}%")
    print("─" * 52)

    # ── Save JSON summary ──────────────────────────────────────────────────
    summary = [
        {"lambda": r["lam"], "test_accuracy": round(r["test_acc"], 2),
         "sparsity_pct": round(r["sparsity"], 2)}
        for r in results
    ]
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Summary saved → results_summary.json]")


if __name__ == "__main__":
    main()
