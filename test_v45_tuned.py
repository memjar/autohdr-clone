"""
AutoHDR Clone v4.5.0 - Fine-tuning Hollywood for balance
Target: Keep brightness ~107, increase saturation to ~50, blacks to ~10%
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
    return {'delta_e': np.mean(delta_e), 'score': correlation * 50 + hist_corr * 50}

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - FINE-TUNING")
    print("Goal: Keep brightness ~107, boost saturation ~50, blacks ~10%")
    print("=" * 70)

    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    print(f"\nTARGET: B=106.8, S=50.4, Blk=10.0%")

    results = []

    # TEST 1: More saturation (via vibrance)
    print("\n" + "-" * 50)
    print("TEST 1: Higher vibrance + less LUT")

    for vibrance in [0.7, 0.8, 0.9]:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.3,
            two_tier_local_strength=0.2,
            use_histogram_params=True,
            hdr_strength=0.35,
            local_contrast=0.18,
            shadow_recovery=0.12,
            highlight_protection=0.5,
            brightness_equalization=False,
            use_clahe=True,
            clahe_clip_limit=1.2,
            auto_dodge_burn=False,
            apply_s_curve=False,
            vibrance=vibrance,
            brightness=-0.22,
            denoise=True,
            denoise_strength=0.45,
            use_hollywood_grading=True,
            use_color_wheels=True,
            use_hollywood_s_curve=True,
            hollywood_shadow_lift=0.06,
            hollywood_midtone_contrast=1.1,
            hollywood_highlight_compress=0.05,
            lut_style='professional_clean',
            lut_intensity=0.2,  # Reduced LUT
        )
        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        print(f"  vibrance={vibrance}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")
        results.append((f"vib={vibrance}", metrics, comp, result))

    # TEST 2: Darker brightness with higher saturation
    print("\n" + "-" * 50)
    print("TEST 2: Brightness adjustment + high vibrance")

    for br in [-0.15, -0.1, -0.05]:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.3,
            two_tier_local_strength=0.2,
            use_histogram_params=True,
            hdr_strength=0.38,
            local_contrast=0.2,
            shadow_recovery=0.15,
            highlight_protection=0.45,
            brightness_equalization=False,
            use_clahe=True,
            clahe_clip_limit=1.3,
            auto_dodge_burn=False,
            apply_s_curve=False,
            vibrance=0.85,
            brightness=br,
            denoise=True,
            denoise_strength=0.45,
            use_hollywood_grading=True,
            use_color_wheels=True,
            use_hollywood_s_curve=True,
            hollywood_shadow_lift=0.05,
            hollywood_midtone_contrast=1.08,
            hollywood_highlight_compress=0.04,
            lut_style='none',  # No LUT
        )
        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        print(f"  brightness={br}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")
        results.append((f"br={br}", metrics, comp, result))

    # TEST 3: Golden Hour for saturation, less intensity
    print("\n" + "-" * 50)
    print("TEST 3: Golden Hour LUT variations")

    for lut_int in [0.2, 0.25, 0.3]:
        settings = ProSettings(
            output_style='natural',
            use_adaptive_processing=True,
            auto_white_balance=True,
            use_two_tier_tone_mapping=True,
            two_tier_global_strength=0.3,
            two_tier_local_strength=0.2,
            use_histogram_params=True,
            hdr_strength=0.38,
            local_contrast=0.2,
            shadow_recovery=0.15,
            highlight_protection=0.45,
            brightness_equalization=False,
            use_clahe=True,
            clahe_clip_limit=1.3,
            auto_dodge_burn=False,
            apply_s_curve=True,
            s_curve_strength=0.15,
            vibrance=0.65,
            brightness=-0.18,
            denoise=True,
            denoise_strength=0.45,
            use_hollywood_grading=True,
            use_color_wheels=False,
            use_hollywood_s_curve=False,
            lut_style='golden_hour',
            lut_intensity=lut_int,
        )
        processor = AutoHDRProProcessor(settings)
        result = processor.process_brackets(brackets)
        metrics = calculate_metrics(result)
        comp = compare_to_target(result, target)
        print(f"  golden_hour={lut_int}: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")
        results.append((f"golden={lut_int}", metrics, comp, result))

    # TEST 4: Balanced - target all metrics
    print("\n" + "-" * 50)
    print("TEST 4: Balanced approach (targeting all 3 metrics)")

    settings_balanced = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.25,
        two_tier_local_strength=0.18,
        use_histogram_params=True,
        hdr_strength=0.4,
        local_contrast=0.22,
        shadow_recovery=0.18,
        highlight_protection=0.4,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.4,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.18,
        vibrance=0.7,
        brightness=-0.12,
        contrast=0.1,
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.01, 0.005),
        use_hollywood_s_curve=False,
        lut_style='golden_hour',
        lut_intensity=0.25,
    )

    processor = AutoHDRProProcessor(settings_balanced)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    print(f"  Balanced: B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")
    results.append(("Balanced", metrics, comp, result))

    # RESULTS
    print("\n" + "=" * 70)
    print("TOP 5 RESULTS")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp, _) in enumerate(results[:5]):
        b_diff = abs(metrics['mean_brightness'] - 106.8)
        s_diff = abs(metrics['saturation'] - 50.4)
        blk_diff = abs(metrics['blacks_pct'] - 10.0)
        total_diff = b_diff + s_diff + blk_diff
        print(f"  {i+1}. {name}: {comp['score']:.1f}%")
        print(f"     B={metrics['mean_brightness']:.1f} (diff {b_diff:.1f}), S={metrics['saturation']:.1f} (diff {s_diff:.1f}), Blk={metrics['blacks_pct']:.1f}% (diff {blk_diff:.1f})")
        print(f"     Total metric diff: {total_diff:.1f}")

    # Save best
    best_name, best_metrics, best_comp, best_result = results[0]
    cv2.imwrite("/tmp/result_v45_best.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved best to /tmp/result_v45_best.jpg")

if __name__ == '__main__':
    main()
