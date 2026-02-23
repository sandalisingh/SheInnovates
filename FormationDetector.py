import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.optimize import linear_sum_assignment
import joblib 
import Configurations as CF

def get_info_for_formation(prediction_string):
    """Safely converts a string to a DB row using Enums"""
    if not isinstance(prediction_string, str):
        return prediction_string
        
    if isinstance(CF.FORMATIONS_INFO_DB, str):
        db = pd.read_csv(CF.FORMATIONS_INFO_DB)
    else:
        db = CF.FORMATIONS_INFO_DB

    parts = prediction_string.split(" ")
    structure = parts[0]
    mode = CF.Mode.BALANCED.value
    shape = CF.Shape.NA.value

    if(len(parts)==2):
        mode = parts[1]
    if(len(parts)==3):
        shape = parts[1]
        mode = parts[2]

    known_modes = [m.value for m in CF.Mode]
    for m in known_modes:
        if m in parts:
            mode = m
            parts.remove(m)
            break

    # match all structure, shape, mode            
    matches = db[
        (db["Structure"] == structure) &
        (db["Mode"].astype(str).str.contains(mode, na=False)) &
        (db["Shape"].astype(str).str.contains(shape, na=False))
    ]
    
    if len(matches) > 0:
        return matches.iloc[0]
    
    # match all structure, mode            
    matches = db[
        (db["Structure"] == structure) &
        (db["Shape"].astype(str).str.contains(shape, na=False))
    ]

    if len(matches) > 0:
        return matches.iloc[0]
    
    # match all structure, mode            
    matches = db[
        (db["Structure"] == structure) &
        (db["Mode"].astype(str).str.contains(mode, na=False)) 
    ]
    
    if len(matches) > 0:
        return matches.iloc[0]
        
    # just look for structure
    matches = db[db["Structure"] == structure]

    if len(matches) > 0:
        return matches.iloc[0]
        
    return None

class FormationDetector:
    def __init__(self, knn_model_path=CF.FORMATION_CLASSIFIER_PATH):
        self.knn_model = joblib.load(knn_model_path)
        self.role_cols = CF.ROLE_VALUE_LIST
        
        # STRICT ENUM MAPPING: 6-tier depth coordinates 
        self.ideal_centroids = {
            CF.Role.Goalkeeper.value: (0.05, 0.5), 
            
            CF.Role.Left_Back.value: (0.18, 0.15), 
            CF.Role.Left_Center_Back.value: (0.18, 0.35), 
            CF.Role.Center_Back.value: (0.18, 0.5), 
            CF.Role.Right_Center_Back.value: (0.18, 0.65), 
            CF.Role.Right_Back.value: (0.18, 0.85), 
            
            CF.Role.Left_Wing_Back.value: (0.30, 0.15), 
            CF.Role.Left_Defensive_Midfielder.value: (0.30, 0.35), 
            CF.Role.Central_Defensive_Midfielder.value: (0.30, 0.5), 
            CF.Role.Right_Defensive_Midfielder.value: (0.30, 0.65),
            CF.Role.Right_Wing_Back.value: (0.30, 0.85),
            
            CF.Role.Left_Midfielder.value: (0.45, 0.15), 
            CF.Role.Left_Central_Midfielder.value: (0.45, 0.35), 
            CF.Role.Central_Midfielder.value: (0.45, 0.5), 
            CF.Role.Right_Central_Midfielder.value: (0.45, 0.65), 
            CF.Role.Right_Midfielder.value: (0.45, 0.85),
            
            CF.Role.Left_Attacking_Midfielder.value: (0.60, 0.35), 
            CF.Role.Central_Attacking_Midfielder.value: (0.60, 0.5), 
            CF.Role.Right_Attacking_Midfielder.value: (0.60, 0.65),
            
            CF.Role.Left_Winger.value: (0.72, 0.15), 
            CF.Role.Left_Forward.value: (0.72, 0.35), 
            CF.Role.Center_Forward.value: (0.72, 0.5), 
            CF.Role.Right_Forward.value: (0.72, 0.65),
            CF.Role.Right_Winger.value: (0.72, 0.85),
            
            CF.Role.Left_Striker.value: (0.82, 0.35), 
            CF.Role.Striker.value: (0.82, 0.5), 
            CF.Role.Right_Striker.value: (0.82, 0.65)
        }

    def normalize_pitch(self, coords):
        coords = coords.copy()
        max_x = np.max(coords[:, 0])
        max_y = np.max(coords[:, 1])
        
        scale_x = max_x if max_x > 105.0 else 105.0
        scale_y = max_y if max_y > 68.0 else 68.0
        
        if scale_x > 0: coords[:, 0] = coords[:, 0] / scale_x
        if scale_y > 0: coords[:, 1] = coords[:, 1] / scale_y
        
        return np.clip(coords, 0, 1)

    def process_centroids(self, df):
        df.columns = df.columns.str.strip().str.upper()
        
        max_x = df['X'].max()
        max_y = df['Y'].max()
        
        # Dynamically scale using standard broadcast dimensions if in pixel space
        if max_x > 105 or max_y > 68: 
            frame_width = max(max_x, 1920.0) 
            frame_height = max(max_y, 1080.0)
            
            df = df.copy() 
            # Normalize to 0.0 - 1.0 range based on image dimensions
            df['X'] = df['X'] / frame_width
            df['Y'] = df['Y'] / frame_height
            
            # Scale up to real pitch meters
            df['X'] = df['X'] * CF.PITCH_LENGTH
            df['Y'] = df['Y'] * CF.PITCH_WIDTH
            
        centroids = df.groupby('ID')[['X', 'Y']].mean().reset_index()
        return centroids[['X', 'Y']].to_numpy(), centroids['ID'].to_numpy()

    def detect_lines(self, players):
        best_labels = None
        best_score = -2  
        
        x_positions = players[:,0].reshape(-1,1)
        x_positions = np.sort(x_positions, axis=0)

        unique_x = len(np.unique(np.round(x_positions, 4)))
        if unique_x < 3:
            return np.zeros(len(players), dtype=int) 

        for k in range(3,6):
            if k > unique_x: break
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(x_positions)
            try:
                score = silhouette_score(x_positions, labels)
            except:
                score = -1

            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None: return np.zeros(len(players), dtype=int)
        return best_labels

    def line_signature(self, labels):
        counts_dict = {}
        for i in range(len(labels)):
            label = labels[i]
            counts_dict[label] = counts_dict.get(label, 0) + 1
        return "-".join(map(str, list(counts_dict.values())))

    def detect_mode(self, players, line_labels=None, team_side="left"):
        players = players.copy()
        if team_side == "right": players[:, 0] = 1 - players[:, 0]

        midfield_line = 0.5
        players_ahead = np.sum(players[:, 0] > midfield_line+0.02)
        players_behind = np.sum(players[:, 0] < midfield_line-0.02)

        if players_ahead > players_behind + 1: return CF.Mode.ATTACKING.value
        if players_behind > players_ahead + 1: return CF.Mode.DEFENDING.value
        return CF.Mode.BALANCED.value

    def coords_to_role_features(self, team_coords_norm):
        role_coords = np.array([self.ideal_centroids[role] for role in self.role_cols])
        cost_matrix = np.linalg.norm(team_coords_norm[:, np.newaxis, :] - role_coords[np.newaxis, :, :], axis=2)
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        feature_vector = np.zeros(len(self.role_cols))
        feature_vector[col_ind] = 1
        return pd.DataFrame([feature_vector], columns=self.role_cols)

    def detect_formation_from_player_positions(self, team_coords, team_side="left"):
        coords = self.normalize_pitch(team_coords)
        if team_side == "right": coords[:,0] = 1 - coords[:,0]

        role_features_df = self.coords_to_role_features(coords)
        predicted_base = self.knn_model.predict(role_features_df)[0]
        
        gk_index = np.argmin(coords[:,0])
        coords_no_gk = np.delete(coords, gk_index, axis=0)
        
        labels = self.detect_lines(coords_no_gk)
        pattern = self.line_signature(labels)
        mode = self.detect_mode(coords_no_gk, labels, team_side)

        final_prediction_name = f"{predicted_base} {mode}"
        return pattern, mode, final_prediction_name