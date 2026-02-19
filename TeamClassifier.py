import numpy as np
from sklearn.cluster import KMeans
import Configurations as CF

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
