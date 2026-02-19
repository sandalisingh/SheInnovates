import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import Configurations as CF
import pandas as pd
import os
from TeamClassifier import TeamClassifier

class ObjectDetector:
    def __init__(self, model_path=CF.YOLO_MODEL_PATH):
        """Initializes the Object Detector and loads the YOLO model ONCE into memory."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"‼️ [ERROR] YOLO model not found at: {model_path}")
        
        print("Loading YOLO Model...")
        self.model = YOLO(model_path)

    def _annotate_and_log(self, frame, box, cls_id, conf, track_id, team_classifier, tracking_data, frame_idx):
        """
        Shared helper method to draw bounding boxes, text, and log coordinates.
        Used by both Image and Video processing to prevent code duplication.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # --- BALL ---
        if cls_id == CF.CLASS_MAP['ball']:
            cv2.rectangle(frame, (x1, y1), (x2, y2), CF.COLORS.get('ball', (0, 255, 255)), 2)

        # --- PLAYERS ---
        elif cls_id == CF.CLASS_MAP['player'] and conf > 0.4:
            team = team_classifier.get_team(frame, box, track_id)
            if team is not None:
                jersey_id = team_classifier.team_to_jersey_map[team][track_id]
                color = CF.COLORS.get(team, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {jersey_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                tracking_data.append([frame_idx, jersey_id, team, cx, cy])

        # --- REFEREE ---
        elif cls_id == CF.CLASS_MAP['referee'] and conf > 0.4:
            cv2.rectangle(frame, (x1, y1), (x2, y2), CF.COLORS.get('referee', (0, 0, 0)), 2)
            cv2.putText(frame, "Ref", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CF.COLORS.get('referee', (0,0,0)), 2)

        # --- GOALKEEPER ---
        elif cls_id == CF.CLASS_MAP['goalkeeper'] and conf > 0.4:
            team = team_classifier.get_team(frame, box, track_id)
            if team is not None:
                jersey_id = team_classifier.team_to_jersey_map[team][track_id]
                color = CF.COLORS.get('goalkeeper', (255, 0, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, "GK", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                tracking_data.append([frame_idx, jersey_id, team, cx, cy])

    def process_video(self, video_path=CF.IP_VID_PATH_OBJ_DET, output_path=CF.OP_VID_PATH_OBJ_DET):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"‼️ [ERROR] Video path does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"‼️ [ERROR] Failed to open video file: {video_path}")
                
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        out    = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        teamClassifier = TeamClassifier()
        tracking_data = []
        
        # --- PHASE 1: Calibration ---
        frames_buffer = []
        for _ in range(60):
            ret, frame = cap.read()
            if not ret: break
            frames_buffer.append(frame)
        
        teamClassifier.train(frames_buffer, self.model)
        
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        print("Starting video processing...")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1, verbose=False)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Video uses persistent tracking IDs
                    if box.id is not None:
                        track_id = int(box.id[0])
                        self._annotate_and_log(frame, box, cls_id, conf, track_id, teamClassifier, tracking_data, frame_idx)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 50 == 0: print(f"Frame {frame_idx} processed")

        cap.release()
        out.release()
        print(f"\nObject detection done! Saved to {output_path}\n")

        return pd.DataFrame(tracking_data, columns=['Frame', 'ID', 'Team', 'X', 'Y'])

    def process_image(self, image_path, output_path=CF.OP_IMG_PATH_OBJ_DET):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"‼️ [ERROR] Image path does not exist: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"‼️ [ERROR] Failed to open image: {image_path}")
            
        teamClassifier = TeamClassifier()
        teamClassifier.train([frame], self.model)
        
        tracking_data = []
        pseudo_track_id = 1 
        accepted_centers = [] # Tracks where we've already placed a player
        
        # Sort boxes by highest confidence first
        results = self.model.predict(frame, conf=0.1, verbose=False)
        
        if results[0].boxes is not None:
            sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf[0]), reverse=True)
            
            for box in sorted_boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Spatial NMS: Ignore duplicate boxes around the same human
                if cls_id == CF.CLASS_MAP['player'] and conf > 0.4:
                    is_duplicate = any(np.sqrt((cx - acx)**2 + (cy - acy)**2) < 40 for acx, acy in accepted_centers)
                    if is_duplicate: continue
                    
                    accepted_centers.append((cx, cy))
                    self._annotate_and_log(frame, box, cls_id, conf, pseudo_track_id, teamClassifier, tracking_data, 0)
                    pseudo_track_id += 1 
                
                elif cls_id in [CF.CLASS_MAP['goalkeeper'], CF.CLASS_MAP['referee']] and conf > 0.4:
                    self._annotate_and_log(frame, box, cls_id, conf, pseudo_track_id, teamClassifier, tracking_data, 0)
                    if cls_id == CF.CLASS_MAP['goalkeeper']: pseudo_track_id += 1

        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            
        df = pd.DataFrame(tracking_data, columns=['Frame', 'ID', 'Team', 'X', 'Y'])
        return df, frame