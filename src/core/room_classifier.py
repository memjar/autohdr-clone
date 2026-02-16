"""
HDRit Room Classifier - Phase 1
================================

Classifies real estate photos into 28 room categories for
room-specific editing profiles.

Phase 1: Rule-based classification using color/brightness analysis
Phase 2: Add ML model (EfficientNet-B5) for accuracy boost
Phase 3: Add YOLO v8 object detection for context awareness
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


class RoomType(Enum):
    """28 Real Estate Room Classifications."""
    # Interior Rooms
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    OFFICE = "office"
    BASEMENT = "basement"
    ATTIC = "attic"
    LAUNDRY = "laundry"
    GARAGE = "garage"
    HALLWAY = "hallway"
    STAIRCASE = "staircase"
    CLOSET = "closet"
    MUDROOM = "mudroom"
    GYM = "gym"
    THEATER = "theater"
    WINE_CELLAR = "wine_cellar"
    SUNROOM = "sunroom"

    # Exterior Spaces
    FRONT_EXTERIOR = "front_exterior"
    BACK_EXTERIOR = "back_exterior"
    SIDE_EXTERIOR = "side_exterior"
    POOL = "pool"
    PATIO = "patio"
    DECK = "deck"
    GARDEN = "garden"
    LANDSCAPE = "landscape"
    DRIVEWAY = "driveway"
    AERIAL = "aerial"


@dataclass
class RoomProfile:
    """Room-specific editing parameters."""
    room_type: RoomType
    brightness_adjust: float  # -1.0 to 1.0
    saturation_adjust: float  # -1.0 to 1.0
    clarity_adjust: float     # 0.0 to 1.0
    color_temp: int           # Kelvin (2700-7500)
    vibrance_adjust: float    # 0.0 to 1.0
    contrast_adjust: float    # -1.0 to 1.0
    shadow_lift: float        # 0.0 to 1.0
    highlight_recovery: float # 0.0 to 1.0


# Pre-defined profiles based on research
ROOM_PROFILES: Dict[RoomType, RoomProfile] = {
    RoomType.KITCHEN: RoomProfile(
        room_type=RoomType.KITCHEN,
        brightness_adjust=0.30,
        saturation_adjust=0.20,
        clarity_adjust=0.40,
        color_temp=5500,
        vibrance_adjust=0.25,
        contrast_adjust=0.15,
        shadow_lift=0.20,
        highlight_recovery=0.30,
    ),
    RoomType.BEDROOM: RoomProfile(
        room_type=RoomType.BEDROOM,
        brightness_adjust=0.10,
        saturation_adjust=0.15,
        clarity_adjust=0.20,
        color_temp=3500,  # Warm and cozy
        vibrance_adjust=0.15,
        contrast_adjust=0.10,
        shadow_lift=0.15,
        highlight_recovery=0.20,
    ),
    RoomType.BATHROOM: RoomProfile(
        room_type=RoomType.BATHROOM,
        brightness_adjust=0.40,
        saturation_adjust=0.10,
        clarity_adjust=0.30,
        color_temp=5000,  # Cool white
        vibrance_adjust=0.10,
        contrast_adjust=0.20,
        shadow_lift=0.25,
        highlight_recovery=0.35,
    ),
    RoomType.LIVING_ROOM: RoomProfile(
        room_type=RoomType.LIVING_ROOM,
        brightness_adjust=0.25,
        saturation_adjust=0.20,
        clarity_adjust=0.30,
        color_temp=4500,
        vibrance_adjust=0.20,
        contrast_adjust=0.15,
        shadow_lift=0.20,
        highlight_recovery=0.25,
    ),
    RoomType.DINING_ROOM: RoomProfile(
        room_type=RoomType.DINING_ROOM,
        brightness_adjust=0.20,
        saturation_adjust=0.25,
        clarity_adjust=0.25,
        color_temp=4000,  # Slightly warm
        vibrance_adjust=0.20,
        contrast_adjust=0.15,
        shadow_lift=0.20,
        highlight_recovery=0.25,
    ),
    RoomType.OFFICE: RoomProfile(
        room_type=RoomType.OFFICE,
        brightness_adjust=0.35,
        saturation_adjust=0.15,
        clarity_adjust=0.35,
        color_temp=5500,  # Daylight
        vibrance_adjust=0.15,
        contrast_adjust=0.20,
        shadow_lift=0.25,
        highlight_recovery=0.30,
    ),
    RoomType.FRONT_EXTERIOR: RoomProfile(
        room_type=RoomType.FRONT_EXTERIOR,
        brightness_adjust=0.20,
        saturation_adjust=0.30,
        clarity_adjust=0.35,
        color_temp=6500,  # Sky blue
        vibrance_adjust=0.40,
        contrast_adjust=0.20,
        shadow_lift=0.30,
        highlight_recovery=0.40,
    ),
    RoomType.BACK_EXTERIOR: RoomProfile(
        room_type=RoomType.BACK_EXTERIOR,
        brightness_adjust=0.20,
        saturation_adjust=0.30,
        clarity_adjust=0.35,
        color_temp=6500,
        vibrance_adjust=0.40,
        contrast_adjust=0.20,
        shadow_lift=0.30,
        highlight_recovery=0.40,
    ),
    RoomType.POOL: RoomProfile(
        room_type=RoomType.POOL,
        brightness_adjust=0.25,
        saturation_adjust=0.35,
        clarity_adjust=0.30,
        color_temp=6000,
        vibrance_adjust=0.45,  # Boost water blue
        contrast_adjust=0.25,
        shadow_lift=0.25,
        highlight_recovery=0.35,
    ),
    RoomType.LANDSCAPE: RoomProfile(
        room_type=RoomType.LANDSCAPE,
        brightness_adjust=0.20,
        saturation_adjust=0.35,
        clarity_adjust=0.40,
        color_temp=6500,
        vibrance_adjust=0.45,  # Boost greens
        contrast_adjust=0.25,
        shadow_lift=0.30,
        highlight_recovery=0.40,
    ),
}

# Default profile for unclassified rooms
DEFAULT_PROFILE = RoomProfile(
    room_type=RoomType.LIVING_ROOM,
    brightness_adjust=0.25,
    saturation_adjust=0.20,
    clarity_adjust=0.30,
    color_temp=5000,
    vibrance_adjust=0.20,
    contrast_adjust=0.15,
    shadow_lift=0.20,
    highlight_recovery=0.25,
)


class RoomClassifier:
    """
    Phase 1: Rule-based room classification using image analysis.

    Strategy:
    1. Analyze color distribution (blue sky = exterior, white tiles = bathroom)
    2. Analyze brightness patterns (windows, lighting)
    3. Analyze edge density (busy kitchen vs minimal bedroom)
    4. Simple heuristics until ML model is added
    """

    def __init__(self):
        self.ml_model = None  # Phase 2: Add EfficientNet-B5
        self.object_detector = None  # Phase 3: Add YOLO v8

    def classify(self, image: np.ndarray) -> Tuple[RoomType, float]:
        """
        Classify room type from image.

        Returns:
            Tuple of (RoomType, confidence 0-1)
        """
        # Phase 1: Heuristic classification
        features = self._extract_features(image)
        room_type, confidence = self._heuristic_classify(features)

        # Phase 2: ML classification (when available)
        if self.ml_model is not None:
            ml_type, ml_conf = self._ml_classify(image)
            if ml_conf > confidence:
                room_type, confidence = ml_type, ml_conf

        return room_type, confidence

    def get_profile(self, room_type: RoomType) -> RoomProfile:
        """Get editing profile for room type."""
        return ROOM_PROFILES.get(room_type, DEFAULT_PROFILE)

    def _extract_features(self, image: np.ndarray) -> Dict:
        """Extract image features for classification."""
        # Color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Brightness
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Edge density (complexity)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Color dominance
        b, g, r = cv2.split(image)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)

        # Sky detection (blue in top portion)
        top_third = image[:image.shape[0]//3, :, :]
        top_hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(top_hsv, (100, 50, 50), (130, 255, 255))
        sky_ratio = np.sum(blue_mask > 0) / blue_mask.size

        # White/tile detection (bathrooms)
        white_mask = cv2.inRange(image, (220, 220, 220), (255, 255, 255))
        white_ratio = np.sum(white_mask > 0) / white_mask.size

        # Green detection (landscaping)
        green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        green_ratio = np.sum(green_mask > 0) / green_mask.size

        return {
            'brightness_mean': np.mean(l_channel),
            'brightness_std': np.std(l_channel),
            'saturation_mean': np.mean(s),
            'edge_density': edge_density,
            'sky_ratio': sky_ratio,
            'white_ratio': white_ratio,
            'green_ratio': green_ratio,
            'avg_r': avg_r,
            'avg_g': avg_g,
            'avg_b': avg_b,
        }

    def _heuristic_classify(self, features: Dict) -> Tuple[RoomType, float]:
        """Rule-based classification from features."""
        scores = {}

        # Exterior detection
        if features['sky_ratio'] > 0.15:
            if features['green_ratio'] > 0.2:
                scores[RoomType.LANDSCAPE] = 0.8
                scores[RoomType.BACK_EXTERIOR] = 0.6
            else:
                scores[RoomType.FRONT_EXTERIOR] = 0.7

        # Pool detection (high blue + some sky)
        if features['sky_ratio'] > 0.1 and features['avg_b'] > features['avg_g']:
            scores[RoomType.POOL] = 0.6

        # Bathroom detection (high white, moderate edges)
        if features['white_ratio'] > 0.15 and features['edge_density'] < 0.15:
            scores[RoomType.BATHROOM] = 0.7

        # Kitchen detection (high edge density, moderate colors)
        if features['edge_density'] > 0.12 and features['brightness_mean'] > 130:
            scores[RoomType.KITCHEN] = 0.6

        # Bedroom detection (low edge density, warm colors)
        if features['edge_density'] < 0.08 and features['saturation_mean'] < 80:
            scores[RoomType.BEDROOM] = 0.6

        # Living room (moderate everything)
        if 0.06 < features['edge_density'] < 0.12:
            scores[RoomType.LIVING_ROOM] = 0.5

        # Default fallback
        if not scores:
            return RoomType.LIVING_ROOM, 0.3

        # Return highest scoring room type
        best_room = max(scores.keys(), key=lambda k: scores[k])
        return best_room, scores[best_room]

    def _ml_classify(self, image: np.ndarray) -> Tuple[RoomType, float]:
        """Phase 2: ML-based classification."""
        # Placeholder for EfficientNet-B5 integration
        # TODO: Load pre-trained model, run inference
        return RoomType.LIVING_ROOM, 0.0


# Convenience function
def classify_room(image: np.ndarray) -> Tuple[RoomType, RoomProfile, float]:
    """
    Classify room and return editing profile.

    Returns:
        Tuple of (RoomType, RoomProfile, confidence)
    """
    classifier = RoomClassifier()
    room_type, confidence = classifier.classify(image)
    profile = classifier.get_profile(room_type)
    return room_type, profile, confidence
