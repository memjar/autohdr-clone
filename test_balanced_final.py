"""
AutoHDR Clone v4.6.0 - Balanced Final Test
Goal: Find the sweet spot between brightness/evenness and saturation
Previous: 74.2% (too dark) vs 52.9% (too bright, washed out)
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
    h, w = gray.shape
    q1, q2, q3, q4 = np.mean(gray[:h//2, :w//2]), np.mean(gray[:h//2, w//2:]), np.mean(gray[h//2:, :w//2]), np.mean(gray[h//2:, w//2:])
    evenness = 1 - np.std([q1, q2, q3, q4]) / np.mean([q1, q2, q3, q4])
    return {
        'mean_brightness': np.mean(gray),
        'saturation': np.mean(hsv[:, :, 1]),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
        'evenness': evenness,
    }

def compare_to_target(result, target):
    if result.shape != target.shape:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result_norm = (result_gray - np.mean(result_gray)) / (np.std(result_gray) + 1e-6)
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    correlation = np.mean(result_norm * target_norm)
    hist_result = cv2.calcHist([result_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_target = cv2.calcHist([target_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist_result, hist_target, cv2.HISTCMP_CORREL)
    return {'score': correlation * 50 + hist_corr * 50}

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - BALANCED FINAL TEST")
    print("Finding sweet spot: brightness=106.8, saturation=50.4")
    print("=" * 70)

    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))

    print("\nTARGET: B=106.8, S=50.4, Even=0.901")

    results = []

    # Moderate CLAHE values to test
    clahe_configs = [
        (1.5, 8),   # Mild
        (1.8, 8),   # Moderate-mild
        (2.0, 8),   # Moderate
        (2.2, 10),  # Moderate-strong
    ]

    brightness_adjusts = [-0.15, -0.1, -0.05, 0.0]

    for clahe_clip, clahe_grid in clahe_configs:
        for br_adj in brightness_adjusts:
            settings = ProSettings(
                output_style='natural',
                use_adaptive_processing=True,
                auto_white_balance=True,
                use_two_tier_tone_mapping=True,
                two_tier_global_strength=0.28,
                two_tier_local_strength=0.18,
                hdr_strength=0.4,
                local_contrast=0.22,
                shadow_recovery=0.25,  # Moderate
                highlight_protection=0.42,
                brightness_equalization=True,
                equalization_strength=0.35,  # Moderate
                use_clahe=True,
                clahe_clip_limit=clahe_clip,
                clahe_grid_size=clahe_grid,
                auto_dodge_burn=True,
                dodge_shadows=0.3,
                burn_highlights=0.1,
                apply_s_curve=True,
                s_curve_strength=0.18,
                vibrance=0.65,  # Strong vibrance to preserve saturation
                brightness=br_adj,
                denoise=True,
                denoise_strength=0.4,
                use_advanced_denoise=True,
                use_perceptual_processing=True,
                use_csf_contrast=True,
                csf_mid_boost=0.1,
                use_hollywood_grading=True,
                use_color_wheels=True,
                use_hollywood_s_curve=False,
                lut_style='golden_hour',
                lut_intensity=0.15,
            )

            processor = AutoHDRProProcessor(settings)
            result = processor.process_brackets(brackets)
            metrics = calculate_metrics(result)
            comp = compare_to_target(result, target)

            name = f"CLAHE={clahe_clip}/br={br_adj}"
            results.append((name, metrics, comp, result))

    # Sort by score
    results.sort(key=lambda x: x[2]['score'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)

    for i, (name, metrics, comp, _) in enumerate(results[:10]):
        b_diff = abs(metrics['mean_brightness'] - 106.8)
        s_diff = abs(metrics['saturation'] - 50.4)
        print(f"  {i+1}. {name}: {comp['score']:.1f}%")
        print(f"     B={metrics['mean_brightness']:.1f} ({b_diff:.1f} off), S={metrics['saturation']:.1f} ({s_diff:.1f} off), Even={metrics['evenness']:.3f}")

    # Save best 3
    for i, (name, metrics, comp, result) in enumerate(results[:3]):
        cv2.imwrite(f"/Users/home/Desktop/autohdr_balanced_{i+1}.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nBest 3 saved to Desktop")

if __name__ == '__main__':
    main()
