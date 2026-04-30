import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys

# Detect environment and set the right backend
def setup_backend():
    env = "unknown"
    try:
        import google.colab
        env = "colab_browser"
    except ImportError:
        pass

    if 'VSCODE_PID' in os.environ or 'VSCODE_CWD' in os.environ:
        env = "vscode"

    print(f"Detected environment: {env}")

    if env == "colab_browser":
        # True Colab browser — widget backend works
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except:
            pass
        matplotlib.use('widget')
    elif env == "vscode":
        # VS Code — use widget (requires ipykernel + ipympl)
        matplotlib.use('widget')
    else:
        # Local Jupyter / fallback
        matplotlib.use('Qt5Agg')

    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    return env

ENV = setup_backend()
print(f"\nOpenCV version: {cv2.__version__}")
print("Setup complete.")


# ────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load images
# ────────────────────────────────────────────────────────────────────────────

# Option A: Upload via Colab file upload (uncomment if in Colab browser)
# from google.colab import files
# uploaded = files.upload()
# C1_PATH = list(uploaded.keys())[0]
# C2_PATH = list(uploaded.keys())[1]

# Option B: Set paths directly (works everywhere including VS Code)
C1_PATH = "c1.jpg"     # ← change to your actual filename/path
C2_PATH = "c2.jpg"    # ← change to your actual filename/path

# Load at 1/4 size as specified in the assignment
im1 = cv2.imread(C1_PATH, cv2.IMREAD_REDUCED_COLOR_4)
im2 = cv2.imread(C2_PATH, cv2.IMREAD_REDUCED_COLOR_4)

if im1 is None: raise FileNotFoundError(f"Cannot find {C1_PATH}")
if im2 is None: raise FileNotFoundError(f"Cannot find {C2_PATH}")

im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
h2, w2  = im2.shape[:2]

print(f"c1 loaded: {im1.shape}")
print(f"c2 loaded: {im2.shape}")

# Quick preview
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(im1_rgb); axes[0].set_title('c1', fontsize=13); axes[0].axis('off')
axes[1].imshow(im2_rgb); axes[1].set_title('c2', fontsize=13); axes[1].axis('off')
plt.suptitle('Study these images before clicking', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q3_preview.png', dpi=120, bbox_inches='tight')
plt.show()
print("Preview saved as q3_preview.png")


# ────────────────────────────────────────────────────────────────────────────
# CELL 3 — Point collection using ginput (works in VS Code with widget backend)
# ────────────────────────────────────────────────────────────────────────────

N_POINTS = 8
POINT_LABELS = [
    "Top-left mounting hole (silver circle, board corner)",
    "DC power jack — top-left corner (black cylinder)",
    "USB connector — top-right corner (silver rectangle)",
    "ATmega2560 IC — top-left corner (large square chip)",
    "ATmega2560 IC — bottom-right corner",
    "Capacitor cluster — centre (two tall silver cylinders)",
    "Red reset button — centre (red circle)",
    "Bottom-left mounting hole (silver circle)",
]

print("=" * 60)
print("  LANDMARK CLICKING GUIDE")
print("=" * 60)
print(f"  Click {N_POINTS} points on EACH image, in this order:")
print()
for i, lbl in enumerate(POINT_LABELS):
    print(f"  {i+1}. {lbl}")
print()
print("  TIP: Use the zoom/pan toolbar to zoom in before clicking.")
print("  TIP: Take your time — accuracy matters.")
print("=" * 60)

# ── Click on c1 ───────────────────────────────────────────────────────────
print(f"\n>>> IMAGE 1 (c1) is opening. Click {N_POINTS} points...")

fig1, ax1 = plt.subplots(figsize=(14, 9))
ax1.imshow(im1_rgb)
ax1.set_title(
    f"c1 — Click {N_POINTS} landmarks IN ORDER (zoom first!)\n"
    f"Figure closes automatically after {N_POINTS} clicks.",
    fontsize=11, color='darkred', fontweight='bold'
)
# Draw reference grid
for x in range(0, im1_rgb.shape[1], 50):
    ax1.axvline(x, color='white', alpha=0.08, lw=0.5)
for y in range(0, im1_rgb.shape[0], 50):
    ax1.axhline(y, color='white', alpha=0.08, lw=0.5)
ax1.axis('off')
plt.tight_layout()

# ginput: blocks until N_POINTS clicks, timeout=300s (5 minutes)
pts1_raw = fig1.ginput(N_POINTS, timeout=300, show_clicks=True)
plt.close(fig1)

pts1 = np.array(pts1_raw, dtype=np.float32)
print(f"\n  Captured {len(pts1)} points on c1:")
for i, p in enumerate(pts1):
    print(f"    {i+1}. ({p[0]:.1f}, {p[1]:.1f})")

# ── Click on c2 ───────────────────────────────────────────────────────────
print(f"\n>>> IMAGE 2 (c2) is opening. Click the SAME {N_POINTS} points in the SAME ORDER...")

fig2, ax2 = plt.subplots(figsize=(14, 9))
ax2.imshow(im2_rgb)
ax2.set_title(
    f"c2 — Click the SAME {N_POINTS} landmarks IN THE SAME ORDER\n"
    f"Figure closes automatically after {N_POINTS} clicks.",
    fontsize=11, color='darkblue', fontweight='bold'
)
for x in range(0, im2_rgb.shape[1], 50):
    ax2.axvline(x, color='white', alpha=0.08, lw=0.5)
for y in range(0, im2_rgb.shape[0], 50):
    ax2.axhline(y, color='white', alpha=0.08, lw=0.5)
ax2.axis('off')
plt.tight_layout()

pts2_raw = fig2.ginput(N_POINTS, timeout=300, show_clicks=True)
plt.close(fig2)

pts2 = np.array(pts2_raw, dtype=np.float32)
print(f"\n  Captured {len(pts2)} points on c2:")
for i, p in enumerate(pts2):
    print(f"    {i+1}. ({p[0]:.1f}, {p[1]:.1f})")

# ── Verify visually ────────────────────────────────────────────────────────
point_colors = plt.cm.tab10(np.linspace(0, 0.9, N_POINTS))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, img, pts, title in zip(
        axes, [im1_rgb, im2_rgb], [pts1, pts2],
        ['c1 — Selected Points', 'c2 — Selected Points']):
    ax.imshow(img)
    for i, (p, c) in enumerate(zip(pts, point_colors)):
        ax.plot(p[0], p[1], 'o', color=c, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.text(p[0]+6, p[1]-8, str(i+1), color=c, fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=1))
    ax.set_title(title, fontsize=12); ax.axis('off')

plt.suptitle('Verify Points — if wrong, re-run CELL 3', fontsize=12,
             color='darkred', fontweight='bold')
plt.tight_layout()
plt.savefig('q3a_points_selected.png', dpi=130, bbox_inches='tight')
plt.show()
print("\nPoint selection complete. Check figure — if correct, run CELL 4.")
print("If points look wrong, re-run CELL 3 to re-click.")


# ────────────────────────────────────────────────────────────────────────────
# CELL 4 — Q3(a): Compute homography + warp
# ────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Q3(a) — Manual Homography")
print("=" * 55)

# Compute H using RANSAC (robust to any slightly imprecise clicks)
H_manual, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
inliers = int(mask_H.ravel().sum()) if mask_H is not None else 0

print(f"\n  Homography matrix H (manual):")
print(H_manual)
print(f"\n  RANSAC inliers: {inliers} / {N_POINTS}")

if inliers < 4:
    print("\n  ⚠️  Too few inliers! Re-run CELL 3 and re-click more carefully.")
else:
    # Warp c1 into perspective of c2
    warped_manual     = cv2.warpPerspective(im1, H_manual, (w2, h2))
    warped_manual_rgb = cv2.cvtColor(warped_manual, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(im1_rgb);           axes[0].set_title('Original c1');         axes[0].axis('off')
    axes[1].imshow(im2_rgb);           axes[1].set_title('Reference c2');        axes[1].axis('off')
    axes[2].imshow(warped_manual_rgb); axes[2].set_title('Warped c1→c2\n(Manual H)'); axes[2].axis('off')
    plt.suptitle('Q3(a) — Manual Homography Warp', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('q3a_warped_manual.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: q3a_warped_manual.png")


# ────────────────────────────────────────────────────────────────────────────
# CELL 5 — Q3(b): Difference image (manual homography)
# ────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Q3(b) — Difference Image (Manual Homography)")
print("=" * 55)

# Float32 conversion prevents uint8 clipping during subtraction
im2_float    = im2_rgb.astype(np.float32)
warped_float = warped_manual_rgb.astype(np.float32)

diff_manual  = np.abs(im2_float - warped_float)

# Mask black (invalid) border pixels produced by warpPerspective
valid_mask = (warped_manual_rgb.sum(axis=2) > 0)
diff_manual[~valid_mask] = 0

# Amplify ×2.5 for visibility
diff_display     = np.clip(diff_manual * 2.5, 0, 255).astype(np.uint8)
mean_diff_manual = diff_manual[valid_mask].mean()

print(f"\n  Mean absolute pixel difference: {mean_diff_manual:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(im2_rgb);           axes[0].set_title('Reference c2');             axes[0].axis('off')
axes[1].imshow(warped_manual_rgb); axes[1].set_title('Warped c1 (Manual H)');     axes[1].axis('off')
axes[2].imshow(diff_display)
axes[2].set_title(f'Difference (×2.5 contrast)\nMean = {mean_diff_manual:.2f} px')
axes[2].axis('off')
plt.suptitle('Q3(b) — Difference Image (Manual Homography)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q3b_diff_manual.png', dpi=130, bbox_inches='tight')
plt.show()
print("Saved: q3b_diff_manual.png")


# ────────────────────────────────────────────────────────────────────────────
# CELL 6 — Q3(c): SIFT detection, matching, display
# ────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Q3(c) — SIFT Feature Detection and Matching")
print("=" * 55)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

print(f"\n  Keypoints — c1: {len(kp1)}   c2: {len(kp2)}")

# FLANN-based matcher with KD-tree index (best for SIFT)
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=50)
)
matches_all  = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test — keeps only unambiguous matches
ratio_thresh = 0.75
good_matches = sorted(
    [m for m, n in matches_all if m.distance < ratio_thresh * n.distance],
    key=lambda x: x.distance
)

print(f"  Raw matches:                     {len(matches_all)}")
print(f"  After Lowe ratio test ({ratio_thresh}):  {len(good_matches)}")

match_img = cv2.drawMatches(
    im1, kp1, im2, kp2,
    good_matches[:60], None,
    matchColor       = (0, 255, 0),
    singlePointColor = (255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(20, 8))
ax.imshow(match_rgb)
ax.set_title(
    f'Q3(c) — SIFT Matches  |  Top 60 of {len(good_matches)} shown\n'
    f'c1: {len(kp1)} kpts   c2: {len(kp2)} kpts   Good: {len(good_matches)}',
    fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('q3c_sift_matches.png', dpi=130, bbox_inches='tight')
plt.show()
print("Saved: q3c_sift_matches.png")


# ────────────────────────────────────────────────────────────────────────────
# CELL 7 — Q3(d): SIFT homography, warp, diff, comparison + discussion
# ────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Q3(d) — SIFT-based Homography")
print("=" * 55)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H_sift, mask_sift = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
inliers_sift      = int(mask_sift.ravel().sum())

print(f"\n  H (SIFT):")
print(H_sift)
print(f"\n  RANSAC inliers: {inliers_sift} / {len(good_matches)}")
print(f"  Inlier ratio  : {inliers_sift/len(good_matches)*100:.1f}%")

warped_sift       = cv2.warpPerspective(im1, H_sift, (w2, h2))
warped_sift_rgb   = cv2.cvtColor(warped_sift, cv2.COLOR_BGR2RGB)
warped_sift_float = warped_sift_rgb.astype(np.float32)
valid_sift        = (warped_sift_rgb.sum(axis=2) > 0)
diff_sift         = np.abs(im2_float - warped_sift_float)
diff_sift[~valid_sift] = 0
diff_sift_disp    = np.clip(diff_sift * 2.5, 0, 255).astype(np.uint8)
mean_diff_sift    = diff_sift[valid_sift].mean()

print(f"\n  Mean diff — Manual H : {mean_diff_manual:.2f} px")
print(f"  Mean diff — SIFT H   : {mean_diff_sift:.2f} px")
print(f"  Improvement          : {mean_diff_manual / mean_diff_sift:.1f}×")

# Figure: SIFT warp + diff
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(im2_rgb);         axes[0].set_title('Reference c2');        axes[0].axis('off')
axes[1].imshow(warped_sift_rgb); axes[1].set_title('Warped c1 (SIFT H)'); axes[1].axis('off')
axes[2].imshow(diff_sift_disp)
axes[2].set_title(f'Difference (×2.5)\nMean = {mean_diff_sift:.2f} px')
axes[2].axis('off')
plt.suptitle('Q3(d) — SIFT Homography: Warp and Difference', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q3d_sift_diff.png', dpi=130, bbox_inches='tight')
plt.show()

# Figure: side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(diff_display)
axes[0].set_title(f'Manual H — Difference\nMean = {mean_diff_manual:.2f} px', fontsize=12)
axes[0].axis('off')
axes[1].imshow(diff_sift_disp)
axes[1].set_title(f'SIFT H — Difference\nMean = {mean_diff_sift:.2f} px', fontsize=12)
axes[1].axis('off')
plt.suptitle('Q3(d) — Comparison: Manual vs SIFT', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q3d_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print("Saved: q3d_sift_diff.png  and  q3d_comparison.png")

print(f"""
==========================================================
  DISCUSSION — Q3(d)
==========================================================

  SIFT H mean diff  = {mean_diff_sift:.2f} px
  Manual H mean diff = {mean_diff_manual:.2f} px
  Improvement        = {mean_diff_manual/mean_diff_sift:.1f}×

  WHY SIFT IS MORE ACCURATE:

  1. CORRESPONDENCES: Manual used {N_POINTS} points; SIFT
     provided {inliers_sift} RANSAC-verified inliers. More
     points = better constrained 8-DOF homography.

  2. PRECISION: Human clicks are ±5–10 px. SIFT
     localises features to sub-pixel accuracy via
     scale-space extrema and quadratic interpolation.

  3. COVERAGE: Manual points cluster near salient
     landmarks. SIFT spreads {len(kp1)}+ keypoints across
     the full board for a more stable DLT solution.

  RESIDUAL ({mean_diff_sift:.2f} px): JPEG artefacts between
  shots, exposure differences, warpPerspective
  interpolation error.

==========================================================
""")
