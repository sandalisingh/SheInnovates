import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from FormationGenerator import FormationGenerator
import Configurations as CF

class FormationDetector:

    def __init__(self, formation_info_df=CF.FORMATIONS_INFO_DB):

        # If user passed dataframe → use directly
        if isinstance(formation_info_df, pd.DataFrame):
            self.formation_info_df = formation_info_df.copy()

        # If user passed path → load CSV
        elif isinstance(formation_info_df, str):
            self.formation_info_df = pd.read_csv(formation_info_df)

        else:
            raise ValueError(
                "FormationDetector expects CSV path or pandas DataFrame"
            )

    def scale_data_to_pitch(self, df):
        """
        CRITICAL FIX: Detects if data is 0-100 and scales it to 
        Pitch Dimensions (105x68).
        """
        # Get data range
        max_x = df['X'].max()
        max_y = df['Y'].max()
        
        # print(f"DEBUG: Data Range found -> X_max: {max_x:.1f}, Y_max: {max_y:.1f}")

        # Heuristic: If Y is > 68 (standard width), it's likely a 0-100 scale
        if max_y > PITCH_WIDTH + 2: 
            # print("⚠️ Scaling Data detected! Converting 0-100 scale to Meters...")
            df = df.copy() # Avoid SettingWithCopyWarning
            
            # Normalize to 0-1
            df['X'] = df['X'] / 100.0
            df['Y'] = df['Y'] / 100.0
            
            # Scale to Meters (105 x 68)
            df['X'] = df['X'] * PITCH_LENGTH
            df['Y'] = df['Y'] * PITCH_WIDTH
            
        return df

    def process_centroids(self, df):
        # 1. Clean Columns
        df.columns = df.columns.str.strip().str.upper()
        
        # 2. Scale Data (The Fix)
        df = self.scale_data_to_pitch(df)
        
        # 3. Calculate Centroids
        centroids = df.groupby('ID')[['X', 'Y']].mean().reset_index()
        return centroids[['X', 'Y']].to_numpy(), centroids['ID'].to_numpy()

    # Normalize pitch coordinates (0 → 1)
    def normalize_pitch(self, coords):

        coords = coords.copy()

        x_range = np.ptp(coords[:, 0])
        y_range = np.ptp(coords[:, 1])

        if x_range > 0:
            coords[:, 0] = (coords[:, 0] - np.min(coords[:, 0])) / x_range

        if y_range > 0:
            coords[:, 1] = (coords[:, 1] - np.min(coords[:, 1])) / y_range

        return coords

    # Detect number of tactical lines automatically
    def detect_lines(self, players):

        best_labels = None
        best_score = -1

        x_positions = players[:,0].reshape(-1,1)

        x_positions = np.sort(x_positions, axis=0)

        # print(f"DEBUG: X Positions for Clustering:\n{x_positions.flatten()}\n")

        # try 3 → 5 lines
        for k in range(3,6):

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(x_positions)

            # silhouette requires >1 sample per cluster
            try:
                score = silhouette_score(x_positions, labels)
            except:
                score = -1

            # print(f"DEBUG: K={k} → Silhouette Score: {score:.4f}")
            # print(f"DEBUG: Cluster Labels for K={k} → {labels}\n")

            if score > best_score:
                best_score = score
                best_labels = labels

        return best_labels

    # Convert cluster labels → formation pattern
    def line_signature(self, labels):
        counts_dict = {}

        for i in range(len(labels)):
            label = labels[i]
            counts_dict[label] = counts_dict.get(label, 0) + 1

        counts = list(counts_dict.values())

        # print(f"DEBUG: Line Counts: {counts}")

        return "-".join(map(str, counts))

    # Detect attacking / defensive / balanced phase
    def detect_mode(self, players, line_labels=None):
        """
        Uses depth, compactness and line structure.
        """

        # Overall team depth
        mean_depth = np.mean(players[:, 0])

        # Defensive line depth (deepest 3 players)
        sorted_x = np.sort(players[:, 0])
        defensive_line_depth = np.mean(sorted_x[:3])

        # Forward line depth (highest 3 players)
        attacking_line_depth = np.mean(sorted_x[-3:])

        # Team vertical stretch
        stretch = np.max(players[:, 0]) - np.min(players[:, 0])

        # -------------------------
        # DECISION LOGIC
        # -------------------------

        # Very high line + stretched
        if mean_depth > 0.6 and attacking_line_depth > 0.75:
            return CF.MODE_ATTACKING

        # Very deep block
        if mean_depth < 0.4 and defensive_line_depth < 0.25:
            return CF.MODE_DEFENSIVE

        # Compact mid-block
        if 0.4 <= mean_depth <= 0.6 and stretch < 0.55:
            return CF.MODE_BALANCED

        # Fallback
        if mean_depth > 0.55:
            return CF.MODE_ATTACKING
        elif mean_depth < 0.45:
            return CF.MODE_DEFENSIVE
        else:
            return CF.MODE_BALANCED

    # MAIN DETECTION FUNCTION
    def detect_formation_from_player_positions(self, team_coords, team_side="left"):

        coords = self.normalize_pitch(team_coords)

        if team_side == "right":
            coords[:,0] = 1 - coords[:,0]

        # remove goalkeeper (deepest player)
        gk_index = np.argmin(coords[:,0])
        coords = np.delete(coords, gk_index, axis=0)

        # print(f"No of coords = {len(coords)}")

        # detect tactical lines
        labels = self.detect_lines(coords)

        pattern = self.line_signature(labels)

        # candidate formations from csv
        matches = self.formation_info_df[
            self.formation_info_df["Formation"].str.contains(pattern, na=False)
        ]

        if len(matches) == 0:
            return pattern, "Unknown", None

        # spatial comparison
        best_matching_formation = None
        best_score = 1e9    

        formationGenerator = FormationGenerator()

        for _, row in matches.iterrows():
            diff_score_from_template = formationGenerator.compare_to_template(coords, row)

            if diff_score_from_template < best_score:
                best_score = diff_score_from_template
                best_matching_formation = row

        mode = self.detect_mode(coords, labels)

        return pattern, mode, best_matching_formation
   
