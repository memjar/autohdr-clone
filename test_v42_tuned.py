"""
AutoHDR Clone v4.2.0 - Tuned to Match Target
Based on target analysis: darker, more saturated, more blacks preserved
"""
import sys
sys.path.insert(0, '/tmp/autohdr-clone/src')

import cv2
import numpy as np
import rawpy
from pathlib import Path
from core.processor_v3 import AutoHDRProProcessor, ProSettings, PROCESSOR_VERSION

ARW_DIR = Path("/Users/home/Downloads/drive-downlo 001 ad-20260205T191553Z-1-")
TARGET_PATH = Path("/Users/home/Downloads/processed_hdr_1770354016632.jpg")

def load_arw(path: Path) -> np.ndarray:
    print(f"  Loading: {path.name}")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def calculate_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    return {
        'mean_brightness': np.mean(gray),
        'contrast': np.std(gray),
        'saturation': np.mean(hsv[:, :, 1]),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
        'shadows_pct': np.sum((L >= 25) & (L < 75)) / L.size * 100,
        'midtones_pct': np.sum((L >= 75) & (L < 180)) / L.size * 100,
        'whites_pct': np.sum(L >= 235) / L.size * 100,
    }

def compare_to_target(result, target):
    if result.shape != target.shape:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_e = np.sqrt(np.sum((result_lab - target_lab)**2, axis=2))
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result_norm = (result_gray - np.mean(result_gray)) / (np.std(result_gray) + 1e-6)
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    correlation = np.mean(result_norm * target_norm)
    hist_result = cv2.calcHist([result_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_target = cv2.calcHist([target_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist_result, hist_target, cv2.HISTCMP_CORREL)
    return {'delta_e': np.mean(delta_e), 'correlation': correlation, 'hist_corr': hist_corr,
            'score': correlation * 50 + hist_corr * 50}

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - TUNED FOR TARGET")
    print("=" * 70)

    # Load
    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    target_metrics = calculate_metrics(target)

    print(f"\nTARGET METRICS (what we're trying to match):")
    print(f"  Brightness: {target_metrics['mean_brightness']:.1f}")
    print(f"  Contrast: {target_metrics['contrast']:.1f}")
    print(f"  Saturation: {target_metrics['saturation']:.1f}")
    print(f"  Blacks: {target_metrics['blacks_pct']:.1f}%")

    results = []

    # ============================================
    # TUNED VERSION 1: Darker, preserve blacks
    # ============================================
    print("\n" + "-" * 50)
    print("TUNED v1: Less brightness equalization, preserve blacks")

    settings_v1 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.45,           # REDUCED from 0.55
        local_contrast=0.25,          # REDUCED
        shadow_recovery=0.25,         # REDUCED significantly
        brightness_equalization=True,
        equalization_strength=0.3,    # REDUCED from 0.5
        use_clahe=True,
        clahe_clip_limit=1.8,         # REDUCED
        auto_dodge_burn=False,        # DISABLED - was lifting too much
        apply_s_curve=True,
        s_curve_strength=0.2,
        vibrance=0.4,                 # INCREASED for saturation
        denoise=True,
        denoise_strength=0.4
    )

    processor_v1 = AutoHDRProProcessor(settings_v1)
    result_v1 = processor_v1.process_brackets(brackets)
    cv2.imwrite("/tmp/result_tuned_v1.jpg", result_v1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_v1 = calculate_metrics(result_v1)
    comp_v1 = compare_to_target(result_v1, target)
    results.append(("Tuned v1", metrics_v1, comp_v1))
    print(f"  Brightness: {metrics_v1['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_v1['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Match Score: {comp_v1['score']:.1f}%")

    # ============================================
    # TUNED VERSION 2: Even darker, more saturation
    # ============================================
    print("\n" + "-" * 50)
    print("TUNED v2: Minimal lifting, max saturation")

    settings_v2 = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,  # Keep it simple
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.35,           # Very reduced
        local_contrast=0.2,
        shadow_recovery=0.15,         # Minimal
        brightness_equalization=False, # DISABLED
        use_clahe=False,              # DISABLED
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.25,
        vibrance=0.6,                 # HIGH
        brightness=-0.2,              # Darken slightly
        denoise=True,
        denoise_strength=0.4
    )

    processor_v2 = AutoHDRProProcessor(settings_v2)
    result_v2 = processor_v2.process_brackets(brackets)
    cv2.imwrite("/tmp/result_tuned_v2.jpg", result_v2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_v2 = calculate_metrics(result_v2)
    comp_v2 = compare_to_target(result_v2, target)
    results.append(("Tuned v2", metrics_v2, comp_v2))
    print(f"  Brightness: {metrics_v2['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_v2['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_v2['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_v2['score']:.1f}%")

    # ============================================
    # TUNED VERSION 3: Target-matched parameters
    # ============================================
    print("\n" + "-" * 50)
    print("TUNED v3: Carefully matched to target")

    settings_v3 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        window_pull='medium',          # Stronger window pull
        hdr_strength=0.4,
        local_contrast=0.22,
        shadow_recovery=0.2,
        highlight_protection=0.4,      # Protect highlights more
        brightness_equalization=True,
        equalization_strength=0.25,    # Very gentle
        use_clahe=True,
        clahe_clip_limit=1.5,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.3,
        contrast=0.15,                 # Add some contrast
        vibrance=0.5,
        brightness=-0.15,              # Slight darken
        denoise=True,
        denoise_strength=0.45
    )

    processor_v3 = AutoHDRProProcessor(settings_v3)
    result_v3 = processor_v3.process_brackets(brackets)
    cv2.imwrite("/tmp/result_tuned_v3.jpg", result_v3, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_v3 = calculate_metrics(result_v3)
    comp_v3 = compare_to_target(result_v3, target)
    results.append(("Tuned v3", metrics_v3, comp_v3))
    print(f"  Brightness: {metrics_v3['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_v3['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_v3['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_v3['score']:.1f}%")

    # ============================================
    # BEST RESULT
    # ============================================
    print("\n" + "=" * 70)
    print("RESULTS RANKING")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp) in enumerate(results):
        print(f"\n  {i+1}. {name}: {comp['score']:.1f}% match")
        print(f"     Delta E: {comp['delta_e']:.1f}, Correlation: {comp['correlation']:.3f}")
        print(f"     Brightness: {metrics['mean_brightness']:.1f}, Saturation: {metrics['saturation']:.1f}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("/tmp/result_tuned_v1.jpg - Less equalization")
    print("/tmp/result_tuned_v2.jpg - Minimal lifting, max saturation")
    print("/tmp/result_tuned_v3.jpg - Carefully matched")

if __name__ == '__main__':
    main()
