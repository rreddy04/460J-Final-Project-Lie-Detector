import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

# Set your input folder path\
#input_folder = "C:\\Users\\Brongus\\Downloads\\testfolder"
input_folder = "C:\\Users\\Brongus\\Downloads\\File gather\\Truths"

# Supported video file extensions
video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')

# Collect all video paths
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
print(f"Found {len(video_files)} video(s)")

def get_video_length_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 if FPS is zero
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps

def process(input_source):
        # region hh
        
        # --- Settings ---
        window_seconds = get_video_length_seconds(input_source)
        blink_threshold = 0.005
        mouth_threshold = 0
        nod_threshold = 0.008

        # --- Setup MediaPipe ---
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_mesh
        pose = mp_pose.Pose()
        face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

        # --- Setup Video ---
        cap = cv2.VideoCapture(input_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames_per_window = int(window_seconds * fps)

        # --- Feature Tracking ---
        movement_scores = []
        head_movements = []
        mouth_activity = []
        blink_count = 0

        window_scores = []
        frame_count = 0
        prev_landmarks = None
        prev_nose = None
        prev_eye_dist = None
        saved_windows = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            # --- Body Movement ---
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                body_points = np.array([
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                    [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                    [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                    [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y],
                ])
                
                if prev_landmarks is not None:
                    diffs = np.linalg.norm(body_points - prev_landmarks, axis=1)
                    jitter_score = np.mean(diffs)
                    movement_scores.append(jitter_score)
                
                prev_landmarks = body_points

                # Head motion
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                nose_pos = np.array([nose.x, nose.y])
                if prev_nose is not None:
                    head_shift = np.linalg.norm(nose_pos - prev_nose)
                    head_movements.append(head_shift)
                prev_nose = nose_pos

            # --- Facial Features ---
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Mouth movement (between upper and lower lips)
                upper_lip = face_landmarks[13].y
                lower_lip = face_landmarks[14].y

                mouth_open_dist = abs(upper_lip - lower_lip)
                face_height = abs(face_landmarks[10].y - face_landmarks[152].y)
                norm_mouth_open = mouth_open_dist / face_height
                
                if norm_mouth_open > mouth_threshold:
                    mouth_activity.append(norm_mouth_open)

                # Blink detection (using eye top and bottom landmarks)
                left_eye_top = face_landmarks[159].y
                left_eye_bottom = face_landmarks[145].y
                eye_dist = abs(left_eye_top - left_eye_bottom)
                if prev_eye_dist is not None and prev_eye_dist - eye_dist > blink_threshold:
                    blink_count += 1
                prev_eye_dist = eye_dist

            frame_count += 1

            # --- Every Window ---

            if frame_count >= frames_per_window:
                magnitude = np.clip(np.mean(movement_scores) * 50, 0, 1) if movement_scores else 0
                nod_shake_count = np.sum(np.array(head_movements) > nod_threshold)
                mouth_score = np.mean(mouth_activity) if mouth_activity else 0
                blink_rate = blink_count / window_seconds
                head_pose_change = np.mean(head_movements) if head_movements else 0

                window_scores.append({
                    'magnitude': magnitude,
                    'nod_shake_rate': nod_shake_count/window_seconds,
                    'nod_shake_count': nod_shake_count,
                    'mouth_activity': mouth_score,
                    'blink_count': blink_count,
                    'blink_rate': blink_rate,
                    'head_pose_change': head_pose_change
                })

                # Reset
                movement_scores = []
                head_movements = []
                mouth_activity = []
                blink_count = 0
                frame_count = 0
                saved_windows+=1

        cap.release()
        
        if saved_windows == 0:
            magnitude = np.clip(np.mean(movement_scores) * 50, 0, 1) if movement_scores else 0
            nod_shake_count = np.sum(np.array(head_movements) > nod_threshold)
            mouth_score = np.mean(mouth_activity) if mouth_activity else 0
            blink_rate = blink_count / (frame_count / fps)
            head_pose_change = np.mean(head_movements) if head_movements else 0

            window_scores.append({
                'magnitude': magnitude,
                'nod_shake_rate': nod_shake_count / (frame_count / fps),
                'nod_shake_count': nod_shake_count,
                'mouth_activity': mouth_score,
                'blink_count': blink_count,
                'blink_rate': blink_rate,
                'head_pose_change': head_pose_change
            })

        # --- Results ---
        for i, s in enumerate(window_scores):
            print(f"Window {i+1}: {s}")
        
        return window_scores
        #endregion

all_video_scores = []

for input_source in video_files:
    print(f"Processing: {input_source}")
    scores = process(input_source)

    video_title = os.path.basename(input_source)

    # Create a dictionary with the video title and full window_scores
    video_row = {
        'video title': video_title,
        'magnitude': [s['magnitude'] for s in scores],
        'nod_shake_rate': [s['nod_shake_rate'] for s in scores],
        'nod_shake_count': [s['nod_shake_count'] for s in scores],
        'mouth_activity': [s['mouth_activity'] for s in scores],
        'blink_count': [s['blink_count'] for s in scores],
        'blink_rate': [s['blink_rate'] for s in scores],
        'head_pose_change': [s['head_pose_change'] for s in scores],
    }

    all_video_scores.append(video_row)

    print("Done")
    print("====================================")
    
# Create a DataFrame where each row has lists of feature values per video
final_df = pd.DataFrame(all_video_scores)

# Save to CSV (optional — note that lists will be stored as strings)
final_df.to_csv("truth_scores.csv", index=False)

print(final_df.head())
    
    
    
    
    