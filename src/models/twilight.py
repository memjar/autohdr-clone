"""
Day-to-Dusk (Virtual Twilight) Pipeline
========================================

Converts daytime exterior photos into twilight/dusk scenes.

Pipeline:
1. Segment sky region
2. Detect windows and light fixtures
3. Replace sky with twilight gradient
4. Add warm glow to windows
5. Add exterior lighting effects
6. Adjust global color temperature

Dependencies:
- segment_anything (SAM) for segmentation
- YOLO or custom model for window detection
- OpenCV for color manipulation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TwilightConfig:
    """Configuration for day-to-dusk conversion."""

    # Sky replacement
    sky_color_top: Tuple[int, int, int] = (20, 30, 60)  # Deep blue (BGR)
    sky_color_horizon: Tuple[int, int, int] = (100, 120, 180)  # Orange/pink (BGR)
    sky_blend_strength: float = 0.9

    # Window glow
    window_glow_color: Tuple[int, int, int] = (50, 140, 255)  # Warm yellow (BGR)
    window_glow_intensity: float = 0.8
    window_glow_blur: int = 15

    # Global adjustments
    color_temp_shift: int = -20  # Shift toward blue
    brightness_reduction: float = 0.7
    saturation_boost: float = 1.1

    # Light spill
    add_light_spill: bool = True
    light_spill_intensity: float = 0.3


class TwilightConverter:
    """
    Convert daytime photos to twilight/dusk.

    Example:
        converter = TwilightConverter()
        twilight = converter.convert(daytime_image)
    """

    def __init__(self, config: Optional[TwilightConfig] = None):
        self.config = config or TwilightConfig()
        self._sky_segmentor = None
        self._window_detector = None

    def _load_sky_segmentor(self):
        """Lazy load SAM for sky segmentation."""
        if self._sky_segmentor is None:
            # TODO: Load Segment Anything or simpler sky detection
            # For now, use color-based detection
            pass

    def segment_sky(self, image: np.ndarray) -> np.ndarray:
        """
        Detect sky region in image.

        Returns binary mask where sky = 255.
        """
        # Simple approach: HSV-based sky detection
        # Works for most outdoor real estate photos
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Sky is typically high brightness, low-mid saturation, blue hue
        # Also detect white/grey overcast sky
        lower_blue = np.array([90, 20, 100])
        upper_blue = np.array([130, 255, 255])

        lower_grey = np.array([0, 0, 150])
        upper_grey = np.array([180, 50, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

        mask = cv2.bitwise_or(mask_blue, mask_grey)

        # Only keep sky in upper portion of image
        h, w = image.shape[:2]
        upper_mask = np.zeros_like(mask)
        upper_mask[:int(h * 0.7), :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)

        return mask

    def detect_windows(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect window regions in image.

        Returns list of bounding boxes (x, y, w, h).

        TODO: Replace with YOLO or trained window detector.
        """
        # Placeholder: edge-based window detection
        # Real implementation would use trained model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find rectangles that could be windows
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        windows = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio and size (windows are typically rectangular)
            aspect = w / max(h, 1)
            area = w * h
            img_area = image.shape[0] * image.shape[1]

            if 0.3 < aspect < 3.0 and 0.001 < area / img_area < 0.1:
                windows.append((x, y, w, h))

        return windows

    def create_twilight_sky(self, size: Tuple[int, int]) -> np.ndarray:
        """Create gradient twilight sky."""
        h, w = size

        # Create vertical gradient
        sky = np.zeros((h, w, 3), dtype=np.uint8)

        for y in range(h):
            ratio = y / h
            # Interpolate between top color and horizon color
            color = [
                int(self.config.sky_color_top[c] * (1 - ratio) +
                    self.config.sky_color_horizon[c] * ratio)
                for c in range(3)
            ]
            sky[y, :] = color

        # Add some variation/clouds
        noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
        for c in range(3):
            sky[:, :, c] = np.clip(sky[:, :, c].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Blur for smoothness
        sky = cv2.GaussianBlur(sky, (21, 21), 0)

        return sky

    def replace_sky(
        self,
        image: np.ndarray,
        sky_mask: np.ndarray,
        new_sky: np.ndarray
    ) -> np.ndarray:
        """Replace sky region with new sky."""
        # Create smooth mask for blending
        mask_float = sky_mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (31, 31), 0)

        # Blend
        result = image.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c] * (1 - mask_float * self.config.sky_blend_strength) +
                new_sky[:, :, c] * mask_float * self.config.sky_blend_strength
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_window_glow(
        self,
        image: np.ndarray,
        windows: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Add warm interior glow to windows."""
        result = image.copy()

        # Create glow layer
        glow_layer = np.zeros_like(image)

        for x, y, w, h in windows:
            # Create elliptical glow
            center = (x + w // 2, y + h // 2)

            cv2.ellipse(
                glow_layer,
                center,
                (w // 2, h // 2),
                0, 0, 360,
                self.config.window_glow_color,
                -1
            )

        # Blur the glow
        glow_layer = cv2.GaussianBlur(glow_layer, (self.config.window_glow_blur * 2 + 1,) * 2, 0)

        # Blend with original
        result = cv2.addWeighted(
            result, 1.0,
            glow_layer, self.config.window_glow_intensity,
            0
        )

        return result

    def adjust_color_temperature(self, image: np.ndarray) -> np.ndarray:
        """Shift image toward cooler (blue) tones for dusk effect."""
        result = image.copy().astype(np.float32)

        # Shift blue channel up, red channel down
        result[:, :, 0] = np.clip(result[:, :, 0] - self.config.color_temp_shift, 0, 255)  # Blue
        result[:, :, 2] = np.clip(result[:, :, 2] + self.config.color_temp_shift // 2, 0, 255)  # Red

        return result.astype(np.uint8)

    def adjust_brightness_saturation(self, image: np.ndarray) -> np.ndarray:
        """Reduce brightness and adjust saturation for dusk."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation_boost, 0, 255)

        # Adjust value (brightness)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.config.brightness_reduction, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def add_light_spill(
        self,
        image: np.ndarray,
        windows: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Add light spill effect on ground/surfaces near windows."""
        if not self.config.add_light_spill:
            return image

        result = image.copy()

        for x, y, w, h in windows:
            # Create light spill below window
            spill_y = y + h
            spill_h = min(h * 2, image.shape[0] - spill_y)

            if spill_h <= 0:
                continue

            # Create gradient
            for dy in range(spill_h):
                ratio = 1 - (dy / spill_h)
                alpha = self.config.light_spill_intensity * ratio * 0.5

                x_start = max(0, x - w // 4)
                x_end = min(image.shape[1], x + w + w // 4)

                for c in range(3):
                    result[spill_y + dy, x_start:x_end, c] = np.clip(
                        result[spill_y + dy, x_start:x_end, c] * (1 - alpha) +
                        self.config.window_glow_color[c] * alpha,
                        0, 255
                    ).astype(np.uint8)

        return result

    def convert(self, image: np.ndarray) -> np.ndarray:
        """
        Full day-to-dusk conversion pipeline.

        Args:
            image: Daytime exterior photo (BGR, uint8)

        Returns:
            Twilight version of the image
        """
        logger.info("Starting day-to-dusk conversion")

        # Step 1: Segment sky
        logger.info("Segmenting sky...")
        sky_mask = self.segment_sky(image)

        # Step 2: Detect windows
        logger.info("Detecting windows...")
        windows = self.detect_windows(image)
        logger.info(f"Found {len(windows)} potential windows")

        # Step 3: Create and replace sky
        logger.info("Replacing sky...")
        twilight_sky = self.create_twilight_sky(image.shape[:2])
        result = self.replace_sky(image, sky_mask, twilight_sky)

        # Step 4: Adjust global color temperature
        logger.info("Adjusting color temperature...")
        result = self.adjust_color_temperature(result)

        # Step 5: Adjust brightness/saturation
        logger.info("Adjusting brightness and saturation...")
        result = self.adjust_brightness_saturation(result)

        # Step 6: Add window glow
        if windows:
            logger.info("Adding window glow...")
            result = self.add_window_glow(result, windows)

            # Step 7: Add light spill
            logger.info("Adding light spill...")
            result = self.add_light_spill(result, windows)

        logger.info("Day-to-dusk conversion complete")
        return result


def main():
    """CLI for day-to-dusk conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert daytime photos to twilight")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--output", "-o", required=True, help="Output image")

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load {args.input}")
        return

    # Convert
    converter = TwilightConverter()
    result = converter.convert(image)

    # Save
    cv2.imwrite(args.output, result)
    print(f"Saved twilight image to {args.output}")


if __name__ == "__main__":
    main()
