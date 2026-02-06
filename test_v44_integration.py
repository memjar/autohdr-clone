"""
AutoHDR Clone v4.4.0 - Integration Test
Tests new features: Two-tier tone mapping + histogram-based parameters
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
        'highlights_pct': np.sum((L >= 180) & (L < 235)) / L.size * 100,
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
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - INTEGRATION TEST")
    print("Testing: Two-tier tone mapping + histogram-based parameters")
    print("=" * 70)

    # Load
    print("\nLoading brackets...")
    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    target_metrics = calculate_metrics(target)

    print(f"\nTARGET METRICS:")
    print(f"  Brightness: {target_metrics['mean_brightness']:.1f}")
    print(f"  Contrast: {target_metrics['contrast']:.1f}")
    print(f"  Saturation: {target_metrics['saturation']:.1f}")
    print(f"  Blacks: {target_metrics['blacks_pct']:.1f}%")

    results = []

    # ============================================
    # TEST 1: New defaults with two-tier + histogram
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 1: v4.4.0 Default (two-tier + histogram)")

    settings_1 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        use_histogram_params=True,
        hdr_strength=0.45,
        local_contrast=0.25,
        shadow_recovery=0.2,
        brightness_equalization=True,
        equalization_strength=0.3,
        use_clahe=True,
        clahe_clip_limit=1.5,
        auto_dodge_burn=False,  # Keep off (was lifting too much)
        apply_s_curve=True,
        s_curve_strength=0.25,
        vibrance=0.5,
        brightness=-0.15,
        denoise=True,
        denoise_strength=0.4
    )

    processor_1 = AutoHDRProProcessor(settings_1)
    result_1 = processor_1.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v44_test1.jpg", result_1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_1 = calculate_metrics(result_1)
    comp_1 = compare_to_target(result_1, target)
    results.append(("v4.4 Default", metrics_1, comp_1))
    print(f"  Brightness: {metrics_1['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_1['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_1['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_1['score']:.1f}%")

    # ============================================
    # TEST 2: Stronger two-tier for more contrast
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 2: Stronger Two-Tier (more local contrast)")

    settings_2 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.5,
        two_tier_local_strength=0.4,
        use_histogram_params=True,
        hdr_strength=0.4,
        local_contrast=0.2,
        shadow_recovery=0.15,
        brightness_equalization=False,  # Let two-tier handle it
        use_clahe=False,  # Avoid double processing
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.3,
        vibrance=0.55,
        brightness=-0.2,
        denoise=True,
        denoise_strength=0.4
    )

    processor_2 = AutoHDRProProcessor(settings_2)
    result_2 = processor_2.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v44_test2.jpg", result_2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_2 = calculate_metrics(result_2)
    comp_2 = compare_to_target(result_2, target)
    results.append(("v4.4 Strong 2-Tier", metrics_2, comp_2))
    print(f"  Brightness: {metrics_2['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_2['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_2['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_2['score']:.1f}%")

    # ============================================
    # TEST 3: Target-matched (based on previous best)
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 3: Target-Matched (previous best + two-tier)")

    settings_3 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.12,
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.2,
        vibrance=0.6,
        brightness=-0.25,
        denoise=True,
        denoise_strength=0.45
    )

    processor_3 = AutoHDRProProcessor(settings_3)
    result_3 = processor_3.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v44_test3.jpg", result_3, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_3 = calculate_metrics(result_3)
    comp_3 = compare_to_target(result_3, target)
    results.append(("v4.4 Target-Matched", metrics_3, comp_3))
    print(f"  Brightness: {metrics_3['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_3['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_3['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_3['score']:.1f}%")

    # ============================================
    # TEST 4: Minimal processing + two-tier only
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 4: Minimal (two-tier primary)")

    settings_4 = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.45,
        two_tier_local_strength=0.35,
        use_histogram_params=True,
        hdr_strength=0.3,  # Very low
        local_contrast=0.15,
        shadow_recovery=0.1,
        brightness_equalization=False,
        use_clahe=False,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.25,
        vibrance=0.65,
        brightness=-0.22,
        denoise=True,
        denoise_strength=0.4
    )

    processor_4 = AutoHDRProProcessor(settings_4)
    result_4 = processor_4.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v44_test4.jpg", result_4, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_4 = calculate_metrics(result_4)
    comp_4 = compare_to_target(result_4, target)
    results.append(("v4.4 Minimal", metrics_4, comp_4))
    print(f"  Brightness: {metrics_4['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_4['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_4['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_4['score']:.1f}%")

    # ============================================
    # COMPARE TO PREVIOUS BEST (65.8%)
    # ============================================
    print("\n" + "-" * 50)
    print("BASELINE: Previous Best (Natural Dark - 65.8%)")

    settings_baseline = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,
        auto_white_balance=True,
        use_two_tier_tone_mapping=False,  # OFF
        use_histogram_params=False,       # OFF
        hdr_strength=0.35,
        shadow_recovery=0.12,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.15,
        vibrance=0.6,
        highlight_protection=0.5,
        brightness=-0.25,
        denoise=True,
        denoise_strength=0.45
    )

    processor_bl = AutoHDRProProcessor(settings_baseline)
    result_bl = processor_bl.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v44_baseline.jpg", result_bl, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_bl = calculate_metrics(result_bl)
    comp_bl = compare_to_target(result_bl, target)
    results.append(("Baseline (65.8%)", metrics_bl, comp_bl))
    print(f"  Brightness: {metrics_bl['mean_brightness']:.1f} (target: {target_metrics['mean_brightness']:.1f})")
    print(f"  Blacks: {metrics_bl['blacks_pct']:.1f}% (target: {target_metrics['blacks_pct']:.1f}%)")
    print(f"  Saturation: {metrics_bl['saturation']:.1f} (target: {target_metrics['saturation']:.1f})")
    print(f"  Match Score: {comp_bl['score']:.1f}%")

    # ============================================
    # RESULTS RANKING
    # ============================================
    print("\n" + "=" * 70)
    print("RESULTS RANKING")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp) in enumerate(results):
        improvement = comp['score'] - 65.8
        imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        print(f"\n  {i+1}. {name}: {comp['score']:.1f}% ({imp_str} vs baseline)")
        print(f"     Delta E: {comp['delta_e']:.1f}, Correlation: {comp['correlation']:.3f}")
        print(f"     Brightness: {metrics['mean_brightness']:.1f}, Saturation: {metrics['saturation']:.1f}")
        print(f"     Blacks: {metrics['blacks_pct']:.1f}%")

    best_name, best_metrics, best_comp = results[0]
    print("\n" + "=" * 70)
    print(f"BEST RESULT: {best_name}")
    print(f"Match Score: {best_comp['score']:.1f}%")
    print("=" * 70)

    print("\nOUTPUT FILES:")
    print("/tmp/result_v44_test1.jpg - v4.4 Default")
    print("/tmp/result_v44_test2.jpg - v4.4 Strong Two-Tier")
    print("/tmp/result_v44_test3.jpg - v4.4 Target-Matched")
    print("/tmp/result_v44_test4.jpg - v4.4 Minimal")
    print("/tmp/result_v44_baseline.jpg - Baseline (previous best)")

if __name__ == '__main__':
    main()
