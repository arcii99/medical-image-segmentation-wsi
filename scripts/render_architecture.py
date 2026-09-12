"""Render the pipeline architecture as a PNG for embedding in the report.

Mirrors docs/architecture.drawio. Coordinates are in the same space so the two
stay visually comparable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

W, H = 1700, 1810
FIG_W = 16.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * H / W), dpi=200)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor("white")

PAL = {
    "in":   ("#F8CECC", "#B85450"),
    "proc": ("#DAE8FC", "#6C8EBF"),
    "gpu":  ("#E1D5E7", "#9673A6"),
    "data": ("#FFF2CC", "#D6B656"),
    "store":("#D5E8D4", "#82B366"),
    "eval": ("#FFE6CC", "#D79B00"),
    "guard":("#F8CECC", "#B85450"),
}
BOXES = {}

def lane(x, y, w, h, label, fc, ec):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.3, linestyle=(0, (6, 4)), zorder=0))
    ax.text(x + 14, y + 20, label, fontsize=11.5, fontweight="bold",
            color=ec, va="top", ha="left", zorder=1)

def box(key, x, y, w, h, title, body="", kind="proc", fs=8.0):
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=8",
                 facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2))
    BOXES[key] = (x, y, w, h)
    cx = x + w / 2
    if body:
        ax.text(cx, y + 13, title, fontsize=fs + 1.1, fontweight="bold",
                ha="center", va="top", zorder=3)
        ax.text(cx, y + 13 + (fs + 7), body, fontsize=fs, ha="center",
                va="top", zorder=3, linespacing=1.45)
    else:
        ax.text(cx, y + h / 2, title, fontsize=fs + 1.1, fontweight="bold",
                ha="center", va="center", zorder=3)

def side(a, b):
    ax_, ay, aw, ah = BOXES[a]; bx, by, bw, bh = BOXES[b]
    acx, acy = ax_ + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    if abs(bcx - acx) >= abs(bcy - acy):
        return ((ax_ + aw, acy), (bx, bcy)) if bcx > acx else ((ax_, acy), (bx + bw, bcy))
    return ((acx, ay + ah), (bcx, by)) if bcy > acy else ((acx, ay), (bcx, by + bh))

def arrow(a, b, label="", color="#4A6B8A", dashed=False, lw=1.9,
          rad=0.0, lx=0, ly=0, fs=7.4):
    (x1, y1), (x2, y2) = side(a, b)
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, zorder=4,
        linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs,
                color=color, ha="center", va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.4))

def poly(pts, label="", color="#4A6B8A", dashed=False, lw=1.9,
         label_at=None, fs=7.4):
    """Arrow along explicit waypoints, so long edges route through empty space."""
    for i in range(len(pts) - 2):
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-",
            linewidth=lw, color=color, zorder=4,
            linestyle="--" if dashed else "-"))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
        mutation_scale=13, linewidth=lw, color=color, zorder=4,
        linestyle="--" if dashed else "-"))
    if label and label_at:
        ax.text(label_at[0], label_at[1], label, fontsize=fs, color=color,
                ha="center", va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=2.0), linespacing=1.4)


def edge_pt(key, where):
    x, y, w, h = BOXES[key]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[where]


def note(x, y, text, color="#B85450", fs=7.6, w=0):
    ax.text(x, y, text, fontsize=fs, color=color, style="italic",
            ha="left", va="top", zorder=3, linespacing=1.5)

# ---------------- title ----------------
ax.text(W / 2, 26, "wsi-metastasis-seg — Gigapixel WSI Tumor Segmentation Pipeline",
        fontsize=17, fontweight="bold", color="#1F4E79", ha="center", va="center")
ax.text(W / 2, 56,
        "Governing constraint: nothing at level-0 resolution is ever materialised in full. "
        "Data moves between modules as COORDINATES; pixels are read lazily and discarded.",
        fontsize=9.4, style="italic", color="#444444", ha="center", va="center")

# ---------------- lanes ----------------
lane(40, 85, 1620, 560, "STAGE A — INGEST   (CPU, offline, once per slide)", "#F4F8FB", "#9BB7CE")
lane(40, 690, 1620, 320, "STAGE B — TRAIN   (GPU, online)", "#F6F4FB", "#B0A3CE")
lane(40, 1062, 1620, 665, "STAGE C — INFER & RECONSTRUCT   (GPU, per slide)", "#F3FAF4", "#95C39B")

# ---------------- stage A ----------------
box("A1", 70, 145, 170, 80, "Raw WSI",
    "pyramidal TIFF / SVS\n≈30 GB uncompressed\n100k × 210k px", "in")
box("A2", 290, 140, 200, 90, "SlideReader",
    "io/slide.py\nbackend chain\nRGBA→RGB on white\nMPP read from file", "proc")
box("A3", 540, 145, 150, 80, "Thumbnail",
    "level ≈ 8 µm/px\n2.8k × 6.6k", "data")
box("A4", 740, 130, 250, 110, "TissueLocalizer",
    "preprocess/tissue.py\n1. artifact mask FIRST (ink, black)\n"
    "2. Otsu(saturation) ∪ Otsu(darkness)\n3. floors: sat ≥ 20, grey ≥ 40\n"
    "4. morphology specified in µm", "proc")
box("A5", 1040, 145, 170, 80, "tissue_mask.png",
    "bool, thumbnail res\nguard 0.01 ≤ mean ≤ 0.90", "data")
box("A6", 70, 290, 170, 80, "Annotation XML",
    "polygon vertices\ngroups _0 / _1 / _2", "in")
box("A7", 290, 285, 230, 100, "AnnotationParser",
    "io/annotations.py\ntumor = union(_0, _1)\ngeom = tumor − union(_2)\n"
    "buffer(0) repairs self-intersections", "proc")
box("A8", 570, 290, 220, 90, "TumorGeometry",
    "VECTOR polygons, level-0 coords\n+ STRtree spatial index\nnever rasterised whole", "data")
box("A9", 740, 455, 250, 105, "PatchIndexer",
    "preprocess/patching.py\ngrid × tissue_frac × tumor_frac\n"
    "tumor_frac by exact POLYGON AREA\n1.2 ms/patch, not 45 min/slide", "proc")
box("A10", 1040, 445, 250, 120, "patches.parquet",
    "slide_id, x0, y0, level, size,\ntissue_frac, tumor_frac, split\n\n"
    "≈40 B/row → 120 MB for 400 slides\n(vs ≈2.1 TB if pixels were dumped)", "store")
box("A11", 1350, 270, 240, 110, "slides.parquet",
    "mpp0, level_w, residual_scale,\npatient_id, split, qc_flag", "store")
note(1350, 400, "Splits assigned HERE, once, by\nhash(patient_id) — never recomputed\nat train time (BUG-007).")

for a, b in [("A1","A2"), ("A2","A3"), ("A3","A4"), ("A4","A5"),
             ("A6","A7"), ("A7","A8"), ("A9","A10")]:
    arrow(a, b)
poly([edge_pt("A5","b"), (1125, 340), (900, 340), edge_pt("A9","t")],
     "tissue gate", "#4A6B8A", label_at=(1010, 330))
poly([edge_pt("A8","b"), (680, 420), (790, 420), edge_pt("A9","l")],
     "tumor_frac", "#4A6B8A", label_at=(742, 405))
arrow("A2", "A7", "", dashed=True, color="#999999", lw=1.3)
poly([edge_pt("A5","r"), (1280, 185), (1280, 325), edge_pt("A11","l")],
     "slide metadata\n+ QC flags", "#82B366", dashed=True, lw=1.5,
     label_at=(1283, 253))

# resolution ladder panel
ax.add_patch(Rectangle((70, 440), 650, 178, facecolor="white",
                       edgecolor="#1F4E79", linewidth=1.3, zorder=2))
ax.text(80, 452, "THE RESOLUTION LADDER — three resolutions in play at once",
        fontsize=8.6, fontweight="bold", color="#1F4E79", va="top", zorder=3)
ax.text(80, 478,
        "Level 0     0.25 µm/px   90k×210k   → COORDINATES ONLY\n"
        "Working     0.50 µm/px   45k×105k   → patch pixels, model input\n"
        "Thumbnail   8.00 µm/px   2.8k×6.6k   → tissue mask, planning, QC",
        fontsize=6.8, family="monospace", va="top", zorder=3, linespacing=1.7)
ax.text(80, 540,
        "INV-1  every persisted coordinate is in LEVEL-0 pixels\n"
        "INV-2  downsample comes from the FILE, never 2^level\n"
        "          true value 1.9992175 → 205 px drift over 105k px\n"
        "read_region(loc, level, size):  loc is LEVEL-0,\n"
        "          size is pixels AT that level",
        fontsize=6.8, family="monospace", va="top", color="#B85450",
        zorder=3, linespacing=1.7)

# ---------------- stage B ----------------
box("B1", 70, 750, 160, 90, "patches.parquet", "coordinates only", "store")
box("B2", 280, 740, 215, 110, "BalancedSampler",
    "data/sampler.py\n1 tumor : 3 normal\n+ 10% boundary bucket\n"
    "+ hard negatives from epoch 8", "gpu")
box("B3", 545, 740, 225, 110, "PatchDataset",
    "data/dataset.py\nLAZY read_region per index\nhandles opened per-PID (BUG-002)\n"
    "±128 px jitter per epoch", "gpu")
box("B4", 820, 740, 255, 110, "Augment",
    "GEOMETRIC D4 → image + mask\nPHOTOMETRIC HED/blur/JPEG\n→ image ONLY (BUG-011)\n"
    "assert mask ∈ {0,1}", "gpu")
box("B5", 1125, 740, 265, 110, "UNet + EfficientNet-B0",
    "6.3 M params, bf16 AMP\n[16,3,512,512] → [16,1,512,512]\n"
    "encoder frozen 2 epochs\npeak ≈5.8 GB of 16 GB", "gpu")
box("B6", 1125, 895, 265, 80, "Loss",
    "0.5·BCE(pos_weight=2)\n+ 0.5·SoftDice (per-image)\nAdamW, cosine + 3ep warmup", "eval")
box("B7", 1440, 878, 200, 115, "best.pt",
    "weights + EMA + optimiser\n+ fitted threshold τ\n+ git SHA + dirty flag\n"
    "+ config + index_hash", "store")
note(70, 890,
     "τ is fitted on validation, NOT 0.5.\n"
     "Balanced sampling distorts the prior,\n"
     "so the cut-off must be re-fitted.\n"
     "τ crosses 5 modules — any one\n"
     "defaulting to 0.5 = 5–10 pt error.")

for a, b in [("B1","B2"), ("B2","B3"), ("B3","B4"), ("B4","B5"),
             ("B5","B6"), ("B6","B7")]:
    arrow(a, b, color="#7B5FA8")
poly([edge_pt("A1","l"), (55, 660), (657, 660), edge_pt("B3","t")],
     "LAZY PIXEL READ  \u2014  the raw WSI is a runtime dependency of training,\n"
     "not just of preprocessing (ADR-005)",
     "#B85450", dashed=True, label_at=(420, 655))

# ---------------- stage C ----------------
box("C1", 70, 1135, 170, 80, "Unseen WSI", "+ tissue mask (reused)", "in")
box("C2", 290, 1120, 255, 120, "SlidingWindowPlanner",
    "infer/sliding_window.py\nstride 256 (50% overlap)\ntissue-gated: 20k of 72k\n"
    "margin CLAMPED, never skipped\nchunk-aligned traversal", "store")
box("C3", 595, 1135, 200, 90, "Batched read_region",
    "batch 32, prefetch thread\noverlaps IO with GPU", "store")
box("C4", 845, 1135, 215, 90, "model + sigmoid",
    "(+ optional 4× D4 TTA)\nprobs [32,1,512,512] ∈ [0,1]", "gpu")
box("C5", 1110, 1115, 290, 130, "GaussianStitcher",
    "infer/stitch.py\nw = exp(−r²/2σ²) + 1e-3,  σ = S/8\n"
    "num += p·w        (FLOAT32)\nden += w             (FLOAT32)\n"
    "accumulated at 8× downsample", "store")
box("C6", 1130, 1305, 250, 105, "Coverage assert",
    "den < 1e-6 inside tissue?\n→ raise CoverageError", "guard")
box("C7", 820, 1315, 240, 85, "Normalise",
    "heat = num / (den + ε)\ncast to float16 ONLY here", "store")
box("C8", 520, 1305, 250, 120, "heatmap.zarr",
    "float16, 2.0 µm/px, 1024² chunks\n148 MB (≈40 MB compressed)\n"
    ".zattrs: mpp, τ, git_sha,\nmodel_run_id, stride", "store")
box("C9", 290, 1500, 250, 140, "Post-processing",
    "infer/postproc.py\nthreshold τ (from checkpoint)\nfill holes\n"
    "drop components < 0.02 mm²\nopening with 10 µm disk\n"
    "ALL in physical units, not px", "store")
box("C10", 600, 1510, 170, 110, "Overlay",
    "jet heatmap α=0.45\nover thumbnail,\nlesion outlines", "data")
box("C11", 830, 1490, 290, 150, "Evaluation",
    "eval/\n1. patch Dice / IoU\n2. FROC (PRIMARY) — sensitivity\n"
    "     @ 0.25/0.5/1/2/4/8 FP per slide\n3. slide-level AUC\n"
    "+ 1000× slide bootstrap CI", "eval")
box("C12", 1180, 1500, 220, 120, "reports/{run_id}/",
    "metrics.json, froc.png,\nper_slide.csv,\nfailure_gallery/", "store")
note(1420, 1315,
     "A HOLE in the heatmap is\nindistinguishable from a confident\n"
     "negative. No metric would reveal it —\nhence a hard assert BEFORE writing.")

for a, b in [("C1","C2"), ("C2","C3"), ("C3","C4"), ("C4","C5"),
             ("C5","C6"), ("C6","C7"), ("C7","C8"), ("C8","C9"),
             ("C9","C10"), ("C11","C12")]:
    arrow(a, b, color="#4E8A57")
poly([edge_pt("C5","t"), (1255, 1101), (695, 1101), edge_pt("C3","t")],
     "loop over every batch", "#999999", dashed=True, lw=1.4,
     label_at=(975, 1098))
poly([edge_pt("C9","b"), (415, 1680), (975, 1680), edge_pt("C11","b")],
     "lesion list", "#D79B00", label_at=(695, 1676))
poly([edge_pt("B7","b"), (1540, 1036), (952, 1036), edge_pt("C4","t")],
     "checkpoint: weights + \u03c4   (the architecture is rebuilt from the CHECKPOINT config,\n"
     "never from the live config file; state_dict loads with strict=True)",
     "#B85450", dashed=True, label_at=(1240, 1033))

# ---------------- legend ----------------
for i, (lab, kind) in enumerate([("process / module", "proc"),
                                 ("persisted artifact", "store"),
                                 ("in-memory data", "data"),
                                 ("guard / hard assert", "guard"),
                                 ("GPU stage", "gpu")]):
    x = 70 + i * 165
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, 1745), 145, 30,
                 boxstyle="round,pad=0,rounding_size=6",
                 facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + 72, 1760, lab, fontsize=8, ha="center", va="center")
ax.text(920, 1760, "──── data flow          - - - -  lazy read / reuse / control",
        fontsize=8, va="center", ha="left")

fig.savefig("/home/claude/architecture.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.12)
print("saved architecture.png")
