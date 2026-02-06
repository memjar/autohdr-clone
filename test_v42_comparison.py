"""
AutoHDR Clone v4.2.0 - Comprehensive Comparison Test
Compares: Previous Version vs Current Version vs Target
"""
import sys
sys.path.insert(0, '/tmp/autohdr-clone/src')

import cv2
import numpy as np
import rawpy
from pathlib import Path
from core.processor_v3 import AutoHDRProProcessor, ProSettings, PROCESSOR_VERSION

# Paths
ARW_DIR = Path("/Users/home/Downloads/drive-downlo 001 ad-20260205T191553Z-1-")
TARGET_PATH = Path("/Users/home/Downloads/processed_hdr_1770354016632.jpg")  # AutoHDR target

def load_arw(path: Path) -> np.ndarray:
    """Load ARW file and convert to BGR"""
    print(f"  Loading: {path.name}")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def calculate_metrics(img, name="Image"):
    """Calculate comprehensive quality metrics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]

    metrics = {
        'mean_brightness': np.mean(gray),
        'std_contrast': np.std(gray),
        'mean_L': np.mean(L),
        'saturation': np.mean(hsv[:, :, 1]),
        'dynamic_range': np.percentile(gray, 95) - np.percentile(gray, 5),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
        'shadows_pct': np.sum((L >= 25) & (L < 75)) / L.size * 100,
        'midtones_pct': np.sum((L >= 75) & (L < 180)) / L.size * 100,
        'highlights_pct': np.sum((L >= 180) & (L < 235)) / L.size * 100,
        'whites_pct': np.sum(L >= 235) / L.size * 100,
    }
    return metrics

def compare_to_target(result, target):
    """Calculate similarity to target image"""
    # Resize if needed
    if result.shape != target.shape:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))

    # Convert to LAB for perceptual comparison
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Delta E (color difference in LAB)
    delta_L = result_lab[:, :, 0] - target_lab[:, :, 0]
    delta_a = result_lab[:, :, 1] - target_lab[:, :, 1]
    delta_b = result_lab[:, :, 2] - target_lab[:, :, 2]

    delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
    mean_delta_e = np.mean(delta_e)

    # Structural similarity (simplified)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Normalized cross-correlation
    result_norm = (result_gray - np.mean(result_gray)) / (np.std(result_gray) + 1e-6)
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    correlation = np.mean(result_norm * target_norm)

    # Histogram similarity
    hist_result = cv2.calcHist([result_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_target = cv2.calcHist([target_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_correlation = cv2.compareHist(hist_result, hist_target, cv2.HISTCMP_CORREL)

    return {
        'delta_e': mean_delta_e,
        'correlation': correlation,
        'histogram_similarity': hist_correlation,
        'match_score': (correlation * 50 + hist_correlation * 50)  # Weighted score
    }

def print_metrics(name, metrics):
    """Print formatted metrics"""
    print(f"\n  {name}:")
    print(f"    Mean Brightness: {metrics['mean_brightness']:.1f}")
    print(f"    Contrast (std): {metrics['std_contrast']:.1f}")
    print(f"    Dynamic Range: {metrics['dynamic_range']:.1f}")
    print(f"    Saturation: {metrics['saturation']:.1f}")
    print(f"    Zone Distribution:")
    print(f"      Blacks (<25):     {metrics['blacks_pct']:5.1f}%")
    print(f"      Shadows (25-75):  {metrics['shadows_pct']:5.1f}%")
    print(f"      Midtones (75-180):{metrics['midtones_pct']:5.1f}%")
    print(f"      Highlights (180+):{metrics['highlights_pct']:5.1f}%")
    print(f"      Whites (235+):    {metrics['whites_pct']:5.1f}%")

def print_comparison(name, comparison):
    """Print target comparison"""
    print(f"\n  {name} vs Target:")
    print(f"    Delta E (color diff): {comparison['delta_e']:.2f} (lower = better)")
    print(f"    Correlation: {comparison['correlation']:.3f} (higher = better)")
    print(f"    Histogram Match: {comparison['histogram_similarity']:.3f}")
    print(f"    MATCH SCORE: {comparison['match_score']:.1f}%")

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - Comprehensive Comparison")
    print("=" * 70)

    # Load brackets
    print("\nLoading ARW brackets...")
    arw_files = sorted(ARW_DIR.glob("*.ARW"))
    if not arw_files:
        print("ERROR: No ARW files found!")
        return

    brackets = [load_arw(f) for f in arw_files]
    print(f"  Loaded {len(brackets)} brackets")

    # Load target
    print("\nLoading target image...")
    if TARGET_PATH.exists():
        target = cv2.imread(str(TARGET_PATH))
        print(f"  Target loaded: {target.shape}")
        has_target = True
    else:
        print("  WARNING: Target image not found, skipping target comparison")
        has_target = False
        target = None

    # ============================================
    # VERSION 1: PREVIOUS (simulated v3.x without new features)
    # ============================================
    print("\n" + "=" * 70)
    print("PROCESSING: Previous Version (v3.x style - no S-curve, no Kuyper)")
    print("=" * 70)

    settings_previous = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,  # Disabled
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        brightness_equalization=True,
        use_clahe=True,
        auto_dodge_burn=True,
        apply_s_curve=False,  # DISABLED
        use_kuyper_masks=False,  # DISABLED
        denoise=True,
        denoise_strength=0.5
    )

    processor_prev = AutoHDRProProcessor(settings_previous)
    result_previous = processor_prev.process_brackets(brackets)
    cv2.imwrite("/tmp/result_previous.jpg", result_previous, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("  Saved: /tmp/result_previous.jpg")

    # ============================================
    # VERSION 2: CURRENT v4.2.0 (all features)
    # ============================================
    print("\n" + "=" * 70)
    print("PROCESSING: Current Version (v4.2.0 - full features)")
    print("=" * 70)

    settings_current = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,  # ENABLED
        adaptive_wb=True,
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        brightness_equalization=True,
        use_clahe=True,
        auto_dodge_burn=True,
        dodge_shadows=0.3,
        burn_highlights=0.15,
        apply_s_curve=True,  # ENABLED
        s_curve_strength=0.25,
        use_kuyper_masks=False,  # Keep off for natural look
        denoise=True,
        denoise_strength=0.5
    )

    processor_curr = AutoHDRProProcessor(settings_current)
    result_current = processor_curr.process_brackets(brackets)
    cv2.imwrite("/tmp/result_current.jpg", result_current, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("  Saved: /tmp/result_current.jpg")

    # ============================================
    # VERSION 3: MAXIMUM (all features cranked up)
    # ============================================
    print("\n" + "=" * 70)
    print("PROCESSING: Maximum Version (all features + Kuyper)")
    print("=" * 70)

    settings_max = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        adaptive_wb=True,
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.6,
        local_contrast=0.32,
        shadow_recovery=0.42,
        brightness_equalization=True,
        equalization_strength=0.6,
        use_clahe=True,
        clahe_clip_limit=2.5,
        auto_dodge_burn=True,
        dodge_shadows=0.35,
        burn_highlights=0.18,
        apply_s_curve=True,
        s_curve_strength=0.3,
        use_kuyper_masks=True,  # ENABLED
        kuyper_lights=-0.1,  # Slight highlight control
        kuyper_darks=0.2,    # Lift shadows
        kuyper_midtones=0.05,
        denoise=True,
        denoise_strength=0.5
    )

    processor_max = AutoHDRProProcessor(settings_max)
    result_max = processor_max.process_brackets(brackets)
    cv2.imwrite("/tmp/result_maximum.jpg", result_max, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("  Saved: /tmp/result_maximum.jpg")

    # ============================================
    # METRICS COMPARISON
    # ============================================
    print("\n" + "=" * 70)
    print("QUALITY METRICS COMPARISON")
    print("=" * 70)

    metrics_prev = calculate_metrics(result_previous)
    metrics_curr = calculate_metrics(result_current)
    metrics_max = calculate_metrics(result_max)

    print_metrics("Previous (v3.x)", metrics_prev)
    print_metrics("Current (v4.2.0)", metrics_curr)
    print_metrics("Maximum (all features)", metrics_max)

    if has_target:
        metrics_target = calculate_metrics(target)
        print_metrics("TARGET", metrics_target)

    # ============================================
    # TARGET COMPARISON
    # ============================================
    if has_target:
        print("\n" + "=" * 70)
        print("TARGET SIMILARITY COMPARISON")
        print("=" * 70)

        comp_prev = compare_to_target(result_previous, target)
        comp_curr = compare_to_target(result_current, target)
        comp_max = compare_to_target(result_max, target)

        print_comparison("Previous (v3.x)", comp_prev)
        print_comparison("Current (v4.2.0)", comp_curr)
        print_comparison("Maximum (all features)", comp_max)

        # Determine winner
        scores = {
            'Previous': comp_prev['match_score'],
            'Current': comp_curr['match_score'],
            'Maximum': comp_max['match_score']
        }
        winner = max(scores, key=scores.get)

        print("\n" + "=" * 70)
        print(f"WINNER: {winner} with {scores[winner]:.1f}% match score!")
        print("=" * 70)

    # ============================================
    # IMPROVEMENT SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY (Previous → Current)")
    print("=" * 70)

    improvements = {
        'Blacks': metrics_prev['blacks_pct'] - metrics_curr['blacks_pct'],
        'Shadows': metrics_prev['shadows_pct'] - metrics_curr['shadows_pct'],
        'Midtones': metrics_curr['midtones_pct'] - metrics_prev['midtones_pct'],
        'Contrast': metrics_curr['std_contrast'] - metrics_prev['std_contrast'],
        'Saturation': metrics_curr['saturation'] - metrics_prev['saturation'],
    }

    for name, value in improvements.items():
        direction = "+" if value > 0 else ""
        quality = "better" if value > 0 else "reduced"
        print(f"  {name}: {direction}{value:.2f} ({quality})")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("/tmp/result_previous.jpg - Previous version (v3.x style)")
    print("/tmp/result_current.jpg  - Current version (v4.2.0)")
    print("/tmp/result_maximum.jpg  - Maximum features enabled")
    if has_target:
        print(f"{TARGET_PATH} - Target to match")

if __name__ == '__main__':
    main()
