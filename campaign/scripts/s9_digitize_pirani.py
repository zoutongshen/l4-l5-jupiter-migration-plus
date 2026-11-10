"""Extract Pirani+2019a's growth tracks from the vector paths of the arXiv PDF (stage 9).

Figs. 1, 3 and 13 are vector graphics: every curve is a coloured polyline, so the
"digitisation" is exact up to the plotting resolution of her figure (no pixel reading).
Page -> data maps come from each panel's frame, whose edges carry the first and last
tick labels (checked against the label positions on the page):

  Fig. 1  (p. 4)  x [68.7, 266.5] pt = 0..30 au,     y [204.6, 76.2] = 1e-2..1e3 M_E (log)
  Fig. 3  (p. 5)  x [335.2, 532.9] = 0..40 au,       same y
  Fig. 13 (p. 11) x [335.2, 532.9] = 1e-1..1e2 M_E (log); y from the label centres,
                  489.1 pt = 1e-1 and 369.4 pt = 1e3 au/Myr (log; the frame's bottom, 498.1, is 10^-1.3)

Legend sample lines (short horizontal segments in the legend box) are dropped.

Figs. 4/5 (raster panels) give the four planets' positions at t = 2.3 ... 3.0 Myr in one of her
nominal simulations: fig4_planets.csv, a direct check of the tracks' clocks.
Solid (core accretion) and dashed (gas accretion) pieces are kept, with a phase flag.

    python campaign/scripts/s9_digitize_pirani.py      # writes campaign/configs/tracks/pirani19a/*.csv
"""

from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "literature" / "02_mechanisms" / "Pirani+2019a_Consequences_of_planetary_migration_on_minor_bodies.pdf"
OUT = Path(__file__).resolve().parents[1] / "configs" / "tracks" / "pirani19a"

# page index, frame (x0, x1, y_bottom, y_top) in pt, data limits, log flags
PANELS = {
    "fig1": dict(page=3, frame=(68.7, 266.5, 204.6, 76.2), x=(0.0, 30.0), y=(1e-2, 1e3), logx=False, logy=True),
    "fig3": dict(page=4, frame=(335.2, 532.9, 204.6, 76.2), x=(0.0, 40.0), y=(1e-2, 1e3), logx=False, logy=True),
    "fig13": dict(page=10, frame=(335.2, 532.9, 498.1, 369.7), ymap=(489.1, 369.4), x=(1e-1, 1e2), y=(1e-1, 1e3),
                  logx=True, logy=True),
}

COLOURS = {
    "fig1": {"jupiter": (1.0, 0.0, 0.0), "saturn": (1.0, 0.65, 0.0), "uranus": (0.0, 0.5, 0.0),
             "neptune": (0.0, 0.0, 1.0)},
    "fig3": {"x0.15": (0.65, 0.16, 0.16), "x0.25": (1.0, 0.0, 0.0), "x0.50": (1.0, 0.65, 0.0),
             "x0.75": (1.0, 1.0, 0.0), "x1.00": (0.0, 1.0, 0.0), "x1.25": (0.0, 0.75, 1.0),
             "x1.50": (0.0, 0.0, 1.0), "x2.00": (0.54, 0.17, 0.89)},
    "fig13": {"nominal": (0.0, 0.0, 1.0), "x0.25": (0.0, 1.0, 0.0), "x0.15": (1.0, 0.0, 0.0),
              "x2.00": (0.58, 0.0, 0.83)},            # v_crit, v_lim are analytic (1.45, 0.17 q Omega r)
}


def _to_data(panel, px, py):
    x0, x1, yb, yt = panel["frame"]
    yb, yt = panel.get("ymap", (yb, yt))
    u = (np.asarray(px) - x0) / (x1 - x0)
    v = (yb - np.asarray(py)) / (yb - yt)
    def lin(f, lim, log):
        return 10 ** (np.log10(lim[0]) + f * (np.log10(lim[1]) - np.log10(lim[0]))) if log else lim[0] + f * (lim[1] - lim[0])
    return lin(u, panel["x"], panel["logx"]), lin(v, panel["y"], panel["logy"])


def _same(c, target, tol=0.02):
    return c is not None and len(c) == 3 and all(abs(a - b) < tol for a, b in zip(c, target))


def extract(name):
    panel = PANELS[name]
    page = pymupdf.open(PDF)[panel["page"]]
    x0, x1, yb, yt = panel["frame"]
    curves = {}
    for label, colour in COLOURS[name].items():
        pts, phase = [], []
        for d in page.get_drawings():
            col = d.get("color")
            if isinstance(colour, str):
                if not (col is not None and max(col) < 0.01 and d.get("width", 0) > 0.8):
                    continue
                dashes = d.get("dashes") or ""
                dash_dot = dashes.count(" ") >= 4           # "[a b c d] 0": dash-dot pattern
                if (colour == "dashdot_black") != dash_dot:
                    continue
            elif not _same(col, colour):
                continue
            seg = [(it[1].x, it[1].y, it[2].x, it[2].y) for it in d["items"] if it[0] == "l"]
            if not seg:
                continue
            xs = np.array([[s[0], s[2]] for s in seg]).ravel()
            ys = np.array([[s[1], s[3]] for s in seg]).ravel()
            inside = (xs > x0 - 0.5) & (xs < x1 + 0.5) & (ys < yb + 0.5) & (ys > yt - 0.5)
            if not inside.any():
                continue
            flat_short = len(seg) == 1 and abs(seg[0][1] - seg[0][3]) < 1e-3 and abs(seg[0][0] - seg[0][2]) < 25
            if flat_short:                                   # a legend sample
                continue
            dashed = bool(d.get("dashes")) and d.get("dashes") not in ("[] 0", "")
            for s in seg:                                    # keep the part inside the frame (curves are clipped)
                for qx, qy in ((s[0], s[1]), (s[2], s[3])):
                    if x0 - 0.5 < qx < x1 + 0.5 and yt - 0.5 < qy < yb + 0.5:
                        pts.append((qx, qy))
                        phase.append(int(dashed))
        if pts:
            px, py = np.array(pts).T
            dx, dy = _to_data(panel, px, py)
            curves[label] = np.column_stack([dx, dy, phase])
    return curves


SNAPSHOT_TIMES_MYR = [2.3, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]       # Figs. 4 and 5, panels in reading order


def extract_snapshots(page_index):
    """Planet positions (the big black dots on the e = 0 / i = 0 axis) in Fig. 4 (p. 6) or Fig. 5 (p. 7).

    These panels are raster images (432 x 288 px, ~11 px/au). The frame is found from its black
    lines (x: 0..30 au), the dots as pure-black blobs in the 14 px above the bottom axis and
    beyond 4.5 au (the asteroid-belt outline is black too); merged dots are split by width.
    Returns rows (t_myr, a_1, ..., a_4), innermost first.
    """
    from PIL import Image
    from scipy import ndimage
    doc = pymupdf.open(PDF)
    page = doc[page_index]
    rows = []
    for k, t in enumerate(SNAPSHOT_TIMES_MYR):
        xref = page.get_image_info(xrefs=True)[k]["xref"]
        pix = pymupdf.Pixmap(doc, xref)
        if pix.alpha or pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        img = np.asarray(Image.frombytes("RGB", (pix.width, pix.height), pix.samples)).astype(int)
        dark = img.max(axis=2) < 80
        cols = np.where(dark.sum(axis=0) > 0.42 * img.shape[0])[0]
        lines = np.where(dark.sum(axis=1) > 0.35 * img.shape[1])[0]
        x0, x1, ybot = cols.min(), cols.max(), lines.max()
        black = img.max(axis=2) < 40
        band = np.zeros_like(black)
        band[ybot - 14:ybot + 1, x0 + 3:x1 - 2] = True
        lab, n = ndimage.label(black & band)
        dots = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < 20 or (xs.mean() - x0) / (x1 - x0) * 30 < 4.5:
                continue
            w = xs.max() - xs.min() + 1
            nd = max(1, int(round(w / 12.5)))
            for j in range(nd):
                sel = (xs >= xs.min() + j * w / nd) & (xs < xs.min() + (j + 1) * w / nd)
                dots.append((xs[sel].mean() - x0) / (x1 - x0) * 30)
        rows.append([t, *sorted(dots)])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f4, f5 = extract_snapshots(5), extract_snapshots(6)
    for r4, r5 in zip(f4, f5):
        assert len(r4) == 5 and np.allclose(r4, r5, atol=0.05), (r4, r5)      # the two figures agree
    np.savetxt(OUT / "fig4_planets.csv", np.array(f4), delimiter=",", fmt="%.3f",
               header="t_myr,a_jupiter,a_saturn,a_uranus,a_neptune  # Pirani+2019a Fig. 4 (= Fig. 5 to 0.05 au), "
                      "one nominal simulation, planet dots", comments="")
    print("fig4_planets.csv:", f4)
    header = {"fig1": "a_au,m_earth,gas_phase", "fig3": "a_au,m_earth,gas_phase", "fig13": "m_earth,adot_au_per_myr,dashed"}
    for name in PANELS:
        for label, arr in extract(name).items():
            path = OUT / f"{name}_{label}.csv"
            np.savetxt(path, arr, delimiter=",", header=header[name] + f"  # Pirani+2019a {name}, {label}, from PDF vectors",
                       fmt=["%.6g", "%.6g", "%d"], comments="")
            print(f"{path.name}: {len(arr)} points, x {arr[:, 0].min():.3g}..{arr[:, 0].max():.3g}, "
                  f"y {arr[:, 1].min():.3g}..{arr[:, 1].max():.3g}, gas/dashed {arr[:, 2].mean():.2f}")


if __name__ == "__main__":
    main()
