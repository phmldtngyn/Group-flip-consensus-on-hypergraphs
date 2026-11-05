"""
Visualization functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


def plot_measurements(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for family in df["family"].unique():
        fam_df = df[df["family"] == family]
        axes[0].scatter(fam_df["q"], fam_df["cG"], label=family, alpha=0.6, s=30)
        axes[1].scatter(fam_df["q"], fam_df["ctilde"], label=family, alpha=0.6, s=30)
    
    axes[0].set_xlabel("q = β/n"); axes[0].set_ylabel("c_G"); axes[0].set_title("Geometric Containment")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("q = β/n"); axes[1].set_ylabel("c̃"); axes[1].set_title("Dynamic Containment")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_exact_1d(records: list, output_path: str):
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    q_vals = np.linspace(0.5, 1.0, 100)
    M_band = (2 - q_vals) / (3 - 2 * q_vals)
    axes[0].plot(q_vals, M_band, 'k-', linewidth=2, label="Band curve")
    
    colors = {'I': 'red', 'II': 'green', 'III': 'blue'}
    for region in df["region"].unique():
        region_df = df[df["region"] == region]
        axes[0].scatter(region_df["q"], region_df["M"], c=colors.get(region, 'gray'), label=f"Region {region}", s=50, alpha=0.7)
        axes[1].scatter(region_df["q"], region_df["M_exact_minus_band"], c=colors.get(region, 'gray'), label=f"Region {region}", s=50, alpha=0.7)
    
    axes[0].set_xlabel("q = β/n"); axes[0].set_ylabel("M"); axes[0].set_title("1D Exact: M vs Band")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
    axes[1].set_xlabel("q = β/n"); axes[1].set_ylabel("M - M_band"); axes[1].set_title("1D Exact: Deviation from Band")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bootstrap(boot_df: pd.DataFrame, output_path: str):
    if len(boot_df) == 0 or "ctilde_coef" not in boot_df.columns:
        print("⚠️ Warning: Bootstrap results are empty. Creating placeholder plot.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Bootstrap failed or returned empty results.\nCheck regression model fit and data variance.", 
                ha='center', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red'))
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(boot_df["ctilde_coef"], bins=30, alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel("c̃ coefficient"); axes[0].set_ylabel("Count"); axes[0].set_title("Bootstrap: c̃ coefficient")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(boot_df["family_norm_reduction"], bins=30, alpha=0.7, edgecolor="black")
    axes[1].axvline(0.3, color='r', linestyle='--', linewidth=2, label="30% threshold")
    axes[1].set_xlabel("Family norm reduction"); axes[1].set_ylabel("Count"); axes[1].set_title("Bootstrap: Family effect reduction")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_family_comparison(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for family in df["family"].unique():
        fam_df = df[df["family"] == family]
        ax.scatter(fam_df["ctilde"], fam_df["q"], label=family, alpha=0.6, s=50)
    ax.set_xlabel("c̃"); ax.set_ylabel("q = β/n"); ax.set_title("Family Comparison: q vs c̃")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()