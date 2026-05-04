import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_two_col_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads COSY text files like BETAX/BETAY/DISPX with a 1-line header and
    two numeric columns: value, element_index.
    Returns (s, value) where s is the element index (float).
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            v = float(parts[0].replace("D", "E"))
            s = float(parts[1].replace("D", "E"))
        except ValueError:
            continue
        data.append((s, v))
    if not data:
        raise ValueError(f"No numeric data parsed from {path}")
    arr = np.array(data, dtype=float)
    # Sort by s in case file isn't ordered
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0], arr[:, 1]


def plot_optics(betax, betay, dispx, title: str, out_png: Path) -> None:
    sx, bx = betax
    sy, by = betay
    sd, dx = dispx

    # Use common x-grid (element index), interpolate to sx grid for consistent plot
    x = sx
    by_i = np.interp(x, sy, by)
    dx_i = np.interp(x, sd, dx)

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    ax.grid(True, alpha=0.3)

    l1, = ax.plot(x, bx, color="red", lw=1.6, label=r"$\beta_x$")
    l2, = ax.plot(x, by_i, color="green", lw=1.6, label=r"$\beta_y$")
    ax.set_ylabel(r"$\beta$ [m]")

    ax2 = ax.twinx()
    l3, = ax2.plot(x, dx_i, color="blue", lw=1.6, label=r"$D_x$")
    ax2.set_ylabel(r"$D$ [m]")

    ax.set_xlabel("element index")
    ax.set_title(title)

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="lower center", ncol=3, frameon=True, framealpha=0.9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat-root", required=True, help="Path to COSY src/dat (contains lattice folders).")
    ap.add_argument("--lattice", required=True, help="Lattice folder name, e.g. magnetic_2p.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--title", default=None, help="Figure title.")
    args = ap.parse_args()

    dat_root = Path(args.dat_root)
    lat_dir = dat_root / args.lattice

    betax = read_two_col_file(lat_dir / "BETAX")
    betay = read_two_col_file(lat_dir / "BETAY")
    dispx = read_two_col_file(lat_dir / "DISPX")

    title = args.title or f"Optics: {args.lattice}"
    plot_optics(betax, betay, dispx, title=title, out_png=Path(args.out))


if __name__ == "__main__":
    main()

