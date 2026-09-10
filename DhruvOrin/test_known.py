"""
DhruvVisionEngine (JarvisVisionEngine) - Face & Person Recognition with Custom Context
=====================================================================================
Detects known people / figures from reference images in `known_faces/`,
counts fingers / hand gestures using MediaPipe, and retrieves custom biographical
profiles from `people_context.json` for personalized AI interaction.

Standalone test:
    python test_known.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# Safe import for face_recognition (dlib)
try:
    import face_recognition
    HAVE_FACE_RECOGNITION = True
except ImportError:
    face_recognition = None
    HAVE_FACE_RECOGNITION = False

# Safe import for mediapipe
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except ImportError:
    mp = None
    HAVE_MEDIAPIPE = False


class JarvisVisionEngine:
    """
    Vision engine for face identification, hand gesture analysis,
    and profile retrieval from JSON for custom conversational responses.
    """

    def __init__(
        self,
        known_faces_dir: str = "known_faces",
        context_json_path: str = "people_context.json",
        tolerance: float = 0.55,
        max_hands: int = 4
    ):
        print("\n[VISION] Initializing Jarvis Vision Engine...")
        self.tolerance = tolerance
        self.max_hands = max_hands

        # Resolve directories (support both current dir and parent repo paths)
        self.faces_dir = self._resolve_path(known_faces_dir)
        self.context_path = self._resolve_path(context_json_path)

        # 1. MediaPipe Hands
        self.hands = None
        self.mp_drawing = None
        self.mp_hands = None
        if HAVE_MEDIAPIPE and mp is not None:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5
                )
                print("[VISION] MediaPipe hand tracking: Initialized.")
            except Exception as e:
                print(f"[VISION Warning]: Failed to init MediaPipe hands: {e}")
        else:
            print("[VISION Notice]: MediaPipe not installed. Hand tracking disabled.")

        # 2. People Context Profiles
        self.people_profiles: Dict[str, str] = {}
        self._load_context_json(self.context_path)

        # 3. Known Face Encodings
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        if HAVE_FACE_RECOGNITION and face_recognition is not None:
            self._load_known_faces(self.faces_dir)
        else:
            print("[VISION Notice]: 'face_recognition' not installed.")
            print("                 Install via: pip install face-recognition")

    @staticmethod
    def _resolve_path(target_path: str) -> Path:
        """Finds target path looking locally first, then in script dir and parent repo."""
        p = Path(target_path)
        if p.exists():
            return p.resolve()

        # Check relative to script directory
        script_dir = Path(__file__).parent.resolve()
        if (script_dir / target_path).exists():
            return (script_dir / target_path).resolve()

        # Check in parent directory
        if (script_dir.parent / target_path).exists():
            return (script_dir.parent / target_path).resolve()

        return (script_dir / target_path).resolve()

    def _load_context_json(self, json_path: Path):
        """Loads contextual biographies / interaction profiles from JSON file."""
        self.people_profiles.clear()
        if not json_path.exists():
            print(f"[VISION Notice]: Context JSON not found at {json_path}. Creating default template.")
            default_template = {
                "Akshat": "Lead engineer and creator of Dhruv. Loves robotics, AI, and innovation. Greet him warmly as your creator!",
                "Anmol": "Core team member and robotics club lead. A passionate builder.",
                "Devendra": "Mentor with extensive experience in IoT, robotics, and embedded systems.",
                "Deepansh": "Mentor and supportive guide for technical development."
            }
            try:
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(default_template, f, indent=4)
                self.people_profiles = default_template
            except Exception as e:
                print(f"[VISION Warning]: Could not create default context JSON: {e}")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Store profiles with both original and lowercase keys for resilient lookup
                    for k, v in data.items():
                        desc = v.get("description", str(v)) if isinstance(v, dict) else str(v)
                        self.people_profiles[k.strip()] = desc.strip()
                    print(f"[VISION] Loaded {len(self.people_profiles)} profile(s) from {json_path.name}")
        except Exception as e:
            print(f"[VISION Warning]: Failed reading context JSON ({json_path}): {e}")

    def _load_known_faces(self, faces_dir: Path):
        """Scans image directory and encodes all reference faces."""
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        if not faces_dir.exists():
            faces_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VISION Notice]: Created known faces folder at {faces_dir}.")
            print("                 Add reference photos (.jpg, .png) of people/figures here.")
            return

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        loaded_count = 0

        for file_path in faces_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    img = face_recognition.load_image_file(str(file_path))
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        name = file_path.stem.capitalize()
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        loaded_count += 1
                        print(f"         -> Loaded face profile: {name} ({file_path.name})")
                    else:
                        print(f"[VISION Warning]: No clear face detected in reference image: {file_path.name}")
                except Exception as e:
                    print(f"[VISION Warning]: Error encoding {file_path.name}: {e}")

        print(f"[VISION] Total known face profiles active: {loaded_count}")

    def get_person_context(self, name: str) -> Optional[str]:
        """Looks up biographical details and custom interaction notes from JSON."""
        if not name or name == "Unknown":
            return None

        # 1. Exact match
        if name in self.people_profiles:
            return self.people_profiles[name]

        # 2. Case-insensitive match
        name_lower = name.lower()
        for k, v in self.people_profiles.items():
            if k.lower() == name_lower:
                return v

        return None

    def _count_fingers(self, hand_landmarks) -> int:
        """Distance-based finger tracking immune to camera mirroring."""
        landmarks = hand_landmarks.landmark
        fingers = 0

        # Thumb Check (Euclidean distance: tip vs knuckle compared to pinky base)
        thumb_tip_dist = (landmarks[4].x - landmarks[17].x) ** 2 + (landmarks[4].y - landmarks[17].y) ** 2
        thumb_base_dist = (landmarks[2].x - landmarks[17].x) ** 2 + (landmarks[2].y - landmarks[17].y) ** 2
        if thumb_tip_dist > thumb_base_dist:
            fingers += 1

        # 4 Fingers Check (Vertical Y coordinate)
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for tip, pip in zip(tip_ids, pip_ids):
            if landmarks[tip].y < landmarks[pip].y:
                fingers += 1

        return fingers

    def process_frame(self, frame) -> Tuple[int, List[str], Dict[str, str], str, np.ndarray]:
        """
        Analyzes a single camera frame for faces, hand gestures, and matching JSON profiles.

        Returns:
            total_fingers (int): Number of fingers held up.
            detected_names (List[str]): Names of detected people (or 'Unknown').
            recognized_contexts (Dict[str, str]): {Name: Context_Bio_From_JSON}.
            visual_summary (str): Formatted string ready for LLM prompt.
            display_frame (np.ndarray): Annotated OpenCV frame for visualization.
        """
        if frame is None:
            return 0, [], {}, "No feed available.", frame

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_names: List[str] = []
        recognized_contexts: Dict[str, str] = {}

        # 1. Multi-Face Recognition
        if HAVE_FACE_RECOGNITION and face_recognition is not None and self.known_face_encodings:
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match = int(np.argmin(face_distances))
                        if matches[best_match]:
                            name = self.known_face_names[best_match]

                    detected_names.append(name)

                    # Lookup custom profile from JSON
                    context = self.get_person_context(name)
                    if context:
                        recognized_contexts[name] = context

                    # Annotate frame
                    is_known = name != "Unknown"
                    box_color = (0, 255, 0) if is_known else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                    label = f"{name}"
                    cv2.putText(display_frame, label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 2. Hand Gesture & Finger Counting (MediaPipe)
        total_fingers = 0
        if self.hands is not None:
            try:
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if self.mp_drawing and self.mp_hands:
                            self.mp_drawing.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        total_fingers += self._count_fingers(hand_landmarks)
            except Exception as e:
                pass

        # 3. Contextual Text Generation for LLM / Chat
        summary_parts = []
        if detected_names:
            known_in_view = [n for n in detected_names if n != "Unknown"]
            if known_in_view:
                summary_parts.append(f"Recognized person(s) in view: {', '.join(known_in_view)}.")
                for kname in known_in_view:
                    if kname in recognized_contexts:
                        summary_parts.append(f"Context for {kname}: {recognized_contexts[kname]}")
            if "Unknown" in detected_names:
                unknown_count = detected_names.count("Unknown")
                summary_parts.append(f"{unknown_count} unrecognized person(s) also present.")
        else:
            summary_parts.append("No human faces detected in current frame.")

        if total_fingers > 0:
            summary_parts.append(f"Hand gesture: A total of {total_fingers} fingers are being held up.")

        visual_summary = " ".join(summary_parts)

        return total_fingers, detected_names, recognized_contexts, visual_summary, display_frame

    def release(self):
        """Releases hardware resources."""
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass


# Backward compatibility alias
KnownFigureEngine = JarvisVisionEngine


# =====================================================================
# STANDALONE TEST RUNNER
# =====================================================================
def run_standalone_test():
    """Runs interactive webcam test showing face recognition and JSON bio lookup."""
    print("=" * 68)
    print("⚡ DHRUV VISION & KNOWN FIGURES TEST RUNNER")
    print("=" * 68)

    engine = JarvisVisionEngine(
        known_faces_dir="known_faces",
        context_json_path="people_context.json",
        tolerance=0.55
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error]: Could not open camera (Index 0). Check connection.")
        return

    print("\n[Controls]: Press 'q' or ESC in preview window to exit.")
    print("            Hold up fingers or show known faces to test.\n")

    last_logged_name = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fingers, names, contexts, summary, display_frame = engine.process_frame(frame)

            # Log to terminal when recognized person appears
            if names and names[0] != "Unknown":
                current_name = names[0]
                if current_name != last_logged_name:
                    last_logged_name = current_name
                    print(f"🎯 [MATCH]: {current_name}")
                    if current_name in contexts:
                        print(f"   📖 [BIO FROM JSON]: {contexts[current_name]}")
                    if fingers > 0:
                        print(f"   🖐️ [FINGERS HELD UP]: {fingers}")
                    print("-" * 50)
            elif not names:
                last_logged_name = None

            # Render summary on display frame
            cv2.putText(
                display_frame,
                f"Fingers: {fingers} | People: {', '.join(names) if names else 'None'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow("Dhruv Vision - Known Figures & Hand Gestures", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.release()
        print("\nVision test closed successfully.")


if __name__ == "__main__":
    run_standalone_test()
