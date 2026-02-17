import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import Configurations as CF
import pandas as pd
import os

class TeamClassifier:
    def __init__(self):
        self.kmeans = None
        self.player_id_to_team = {} # Stores {track_id: team_label}
        self.team_to_jersey_map = {'team_0': {}, 'team_1': {}}
        self.next_jersey_number = {'team_0': 1, 'team_1': 1}

    def get_shirt_crop(self, frame, box):
        """
        Extracts only the 'shirt' area (top center) to avoid
        background grass and shorts mixing into the color.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h = y2 - y1
        w = x2 - x1
        
        # KEY FIX: Crop stricter. 
        # Top 15% to 50% (The chest area)
        # Central 40% width (Avoid grass on sides)
        y_start = y1 + int(h * 0.15)
        y_end   = y1 + int(h * 0.50)
        x_start = x1 + int(w * 0.30)
        x_end   = x2 - int(w * 0.30)
        
        # Safety check to keep within frame
        h_img, w_img, _ = frame.shape
        y_start, y_end = max(0, y_start), min(h_img, y_end)
        x_start, x_end = max(0, x_start), min(w_img, x_end)
        
        return frame[y_start:y_end, x_start:x_end]

    def train(self, frames_buffer, model):
        """
        Learns the two dominant team colors from the first 60 frames.
        """
        print("Training Team Classifier (Sampling shirt colors)...")
        player_colors = []
        
        # Process the buffered frames
        for frame in frames_buffer:
            results = model(frame, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) == CF.CLASS_MAP['player']:
                    crop = self.get_shirt_crop(frame, box)
                    if crop.size > 0:
                        # Get average color of the SHIRT only
                        avg_color = np.mean(crop, axis=(0, 1))
                        player_colors.append(avg_color)
            
        if len(player_colors) > 0:
            # Force 2 clusters (Team A vs Team B)
            self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            self.kmeans.fit(player_colors)
            print("Done. Team colors learned.")

    def get_team(self, frame, box, track_id):
        # 1. No Flickering: If ID is known, return saved team
        if track_id in self.player_id_to_team:
            return self.player_id_to_team[track_id]

        # 2. Get Shirt Color
        crop = self.get_shirt_crop(frame, box)
        if crop.size == 0: return 'team_0' # Fallback

        avg_color = np.mean(crop, axis=(0, 1)).reshape(1, -1)
        
        # 3. Predict
        label = self.kmeans.predict(avg_color)[0]
        team_name = f'team_{label}'

        # Assign jersey number (1–11 per team)
        if track_id not in self.team_to_jersey_map[team_name]:
            jersey = self.next_jersey_number[team_name]
            if jersey <= 11:
                self.team_to_jersey_map[team_name][track_id] = jersey
                self.next_jersey_number[team_name] += 1
            else:
                self.team_to_jersey_map[team_name][track_id] = 11

        self.player_id_to_team[track_id] = team_name
        return team_name

def ObjectDetection(video_path=CF.IP_VID_PATH_OBJ_DET, output_path=CF.OP_VID_PATH_OBJ_DET):
    # ------------------ Check input video path ------------------
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"‼️ [ERROR] Video path does not exist: {video_path}")

    if not os.path.isfile(video_path):
        raise ValueError(f"‼️ [ERROR] Provided path is not a file: {video_path}")

    # ------------------ Check YOLO model path ------------------
    if not os.path.exists(CF.YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"‼️ [ERROR] YOLO model not found at: {CF.YOLO_MODEL_PATH}"
        )
    model = YOLO(CF.YOLO_MODEL_PATH)

    # ------------------ Try opening video ------------------
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"‼️ [ERROR] Failed to open video file: {video_path}"
        )
            
    tracking_data = []
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    out    = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    teamClassifier = TeamClassifier()
    
    # --- PHASE 1: Calibration ---
    # Read 60 frames to learn colors
    frames_buffer = []
    for _ in range(60):
        ret, frame = cap.read()
        if not ret: break
        frames_buffer.append(frame)
    
    # Train the classifier on these frames
    teamClassifier.train(frames_buffer, model)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    print("Starting processing...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # High-sensitivity tracking for Ball (conf=0.1), stricter for players later
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1, verbose=False)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # --- BALL ---
                if cls_id == CF.CLASS_MAP['ball']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CF.COLORS['ball'], 2)
                    
                # --- PLAYERS ---
                elif cls_id == CF.CLASS_MAP['player']:
                    # Only accept players with decent confidence
                    if conf > 0.4 and box.id is not None:
                        track_id = int(box.id[0])
                        team = teamClassifier.get_team(frame, box, track_id)
                        if team is None:
                            continue

                        jersey_id = teamClassifier.team_to_jersey_map[team][track_id]
                        color = CF.COLORS[team]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Compute center
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        tracking_data.append([frame_idx, jersey_id, team, cx, cy])

                # --- REF & GK ---
                elif cls_id == CF.CLASS_MAP['referee'] and conf > 0.4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CF.COLORS['referee'], 2)
                    cv2.putText(frame, "Ref", (x1, y1-5), 0, 0.5, CF.COLORS['referee'], 2)
                
                elif cls_id == CF.CLASS_MAP['goalkeeper'] and conf > 0.4 and box.id is not None:
                    track_id = int(box.id[0])
                    team = teamClassifier.get_team(frame, box, track_id)
                    if team is None:
                        continue

                    jersey_id = teamClassifier.team_to_jersey_map[team][track_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CF.COLORS['goalkeeper'], 2)
                    cv2.putText(frame, "GK", (x1, y1-5), 0, 0.5, CF.COLORS['goalkeeper'], 2)

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    tracking_data.append([frame_idx, jersey_id, team, cx, cy])

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: print(f"Frame {frame_idx} processed")

    cap.release()
    out.release()
    print("\nObject detection done!\n")

    df = pd.DataFrame(tracking_data, columns=['Frame', 'ID', 'Team', 'X', 'Y'])
    return df

