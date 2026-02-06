"""
AutoHDR Clone v4.4.0 - Brightness Tuning
Focus: Get brightness closer to 106.8 while keeping saturation/blacks good
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
        'saturation': np.mean(hsv[:, :, 1]),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
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
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - BRIGHTNESS TUNING")
    print("Target brightness: 106.8, saturation: 50.4, blacks: 10.0%")
    print("=" * 70)

    # Load
    print("\nLoading brackets...")
    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))

    results = []

    # Test different brightness settings
    brightness_values = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    for br in brightness_values:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.35,
            two_tier_local_strength=0.25,
            use_histogram_params=True,
            hdr_strength=0.4,  # Slightly higher for brightness
            local_contrast=0.2,
            shadow_recovery=0.18,  # More shadow recovery
            highlight_protection=0.45,
            brightness_equalization=True,
            equalization_strength=0.25,  # Light equalization
            use_clahe=True,
            clahe_clip_limit=1.3,
            auto_dodge_burn=False,
            apply_s_curve=True,
            s_curve_strength=0.2,
            vibrance=0.55,
            brightness=br,
            denoise=True,
            denoise_strength=0.4
        )

        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        results.append((f"br={br:.2f}", metrics, comp, result))
        print(f"  brightness={br:+.2f}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}%, Score={comp['score']:.1f}%")

    # Find best
    results.sort(key=lambda x: x[2]['score'], reverse=True)
    best_name, best_metrics, best_comp, best_result = results[0]

    print(f"\nBest: {best_name} with {best_comp['score']:.1f}%")

    # Save best
    cv2.imwrite("/tmp/result_v44_bright_tuned.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Now try with more shadow recovery to lift overall brightness
    print("\n" + "-" * 50)
    print("Testing shadow recovery variations...")

    for sr in [0.2, 0.25, 0.3, 0.35]:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.35,
            two_tier_local_strength=0.25,
            use_histogram_params=True,
            hdr_strength=0.42,
            local_contrast=0.22,
            shadow_recovery=sr,
            highlight_protection=0.4,
            brightness_equalization=True,
            equalization_strength=0.3,
            use_clahe=True,
            clahe_clip_limit=1.4,
            auto_dodge_burn=False,
            apply_s_curve=True,
            s_curve_strength=0.22,
            vibrance=0.52,
            brightness=0.0,
            denoise=True,
            denoise_strength=0.4
        )

        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        print(f"  shadow_recovery={sr:.2f}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}%, Score={comp['score']:.1f}%")
        if comp['score'] > best_comp['score']:
            best_name = f"sr={sr}"
            best_metrics = metrics
            best_comp = comp
            best_result = result

    # Try equalization strength
    print("\n" + "-" * 50)
    print("Testing equalization strength...")

    for eq in [0.35, 0.4, 0.45, 0.5]:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.35,
            two_tier_local_strength=0.25,
            use_histogram_params=True,
            hdr_strength=0.42,
            local_contrast=0.22,
            shadow_recovery=0.22,
            highlight_protection=0.4,
            brightness_equalization=True,
            equalization_strength=eq,
            use_clahe=True,
            clahe_clip_limit=1.5,
            auto_dodge_burn=False,
            apply_s_curve=True,
            s_curve_strength=0.22,
            vibrance=0.5,
            brightness=0.0,
            denoise=True,
            denoise_strength=0.4
        )

        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        print(f"  equalization={eq:.2f}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}%, Score={comp['score']:.1f}%")
        if comp['score'] > best_comp['score']:
            best_name = f"eq={eq}"
            best_metrics = metrics
            best_comp = comp
            best_result = result

    print("\n" + "=" * 70)
    print(f"BEST OVERALL: {best_name}")
    print(f"  Brightness: {best_metrics['mean_brightness']:.1f} (target: 106.8)")
    print(f"  Saturation: {best_metrics['saturation']:.1f} (target: 50.4)")
    print(f"  Blacks: {best_metrics['blacks_pct']:.1f}% (target: 10.0%)")
    print(f"  Match Score: {best_comp['score']:.1f}%")
    print("=" * 70)

    cv2.imwrite("/tmp/result_v44_best.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("\nSaved: /tmp/result_v44_best.jpg")

if __name__ == '__main__':
    main()
