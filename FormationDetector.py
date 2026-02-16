import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import numpy as np

FORMATIONS_INFO_DB = pd.read_csv("Data/Formations_info.csv")

PITCH_LENGTH = 105
PITCH_WIDTH = 68
PITCH_COLOUR = '#2e8b57'

AREA_CAPTURE_MIN_THRESHOLD = 550 # in m^2

class FormationDetector:

    def __init__(self, formation_info_df=FORMATIONS_INFO_DB):

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

        # role columns used for template matching
        self.role_columns = [
            "LCB", # Left Center Back
            "CB", # Center Back
            "RCB", # Right Center Back
            "LB", # Left Back
            "RB", # Right Back
            "CDM", # Defensive Midfielder
            "CM", # Central Midfielder
            "CAM", # Central Attacking Midfielder
            "LM", # Left Midfielder
            "RM", # Right Midfielder
            "LW", # Left Winger
            "RW", # Right Winger
            "ST", # Striker
            "CF", # Center Forward
            "LS", # Left Striker
            "RS"  # Right Striker
        ]

        self.role_columns = [
            c for c in self.role_columns if c in self.formation_info_df.columns
        ]

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
            print("⚠️ Scaling Data detected! Converting 0-100 scale to Meters...")
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

        print(f"DEBUG: X Positions for Clustering:\n{x_positions.flatten()}\n")

        # try 3 → 5 lines
        for k in range(3,6):

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(x_positions)

            # silhouette requires >1 sample per cluster
            try:
                score = silhouette_score(x_positions, labels)
            except:
                score = -1

            print(f"DEBUG: K={k} → Silhouette Score: {score:.4f}")
            print(f"DEBUG: Cluster Labels for K={k} → {labels}\n")

            if score > best_score:
                best_score = score
                best_labels = labels

        return best_labels

    # Convert cluster labels → formation pattern
    def line_signature(self, labels):

        counts = [np.sum(labels == i) for i in np.unique(labels)]
        counts.sort(reverse=True)

        return "-".join(map(str, counts))

    # Create template positions from CSV row
    def build_template_for_formation_info(self, formation_row):
        formation_row = formation_row.copy() 
        # print(f"DEBUG: Building template for formation row:\n{formation_row}\n")

        template = []

        # depth levels (def → attack)
        depth_map = {
            "B": 0.2,   # Backline
            "D": 0.35,  # Defensive Midfield
            "M": 0.55,  # Midfield
            "A": 0.75,  # Attacking Midfield
            "F": 0.85   # Forwards
        }

        x_coords = []

        for role in self.role_columns:

            if role not in formation_row:
                continue

            count = int(formation_row[role])
            if count == 0:
                continue

            for i in range(count):

                if "CB" in role or "LB" in role or "RB" in role:
                    x = depth_map["D"]
                elif "CDM" in role:
                    x = depth_map["M"] - 0.1
                elif "CM" in role or "LM" in role or "RM" in role:
                    x = depth_map["M"]
                elif "CAM" in role:
                    x = depth_map["A"]
                else:
                    x = depth_map["F"]

                x_coords.append(x)

        x_count = pd.Series(x_coords).value_counts().to_dict()
        
        # assign y based on same x
        for x, count in x_count.items():
            y_positions = np.linspace(0.0, 1.0, count + 2)[1:-1] # avoid edges
            for y in y_positions:
                template.append([x, y])

        # print(f"DEBUG: Built template with {player_count} outfield players:\n{np.array(template)}\n")

        return np.array(template)

    # Compare real players to formation template
    def compare_to_template(self, players, formation_row):

        template = self.build_template_for_formation_info(formation_row)

        if len(template) == 0:
            return 1e9

        # match sizes
        n = min(len(players), len(template))
        players = players[:n]
        template = template[:n]

        cost = cdist(players, template)

        row_ind, col_ind = linear_sum_assignment(cost)

        return cost[row_ind, col_ind].mean()

    # Detect attacking / defensive / balanced phase
    def detect_mode(self, players):

        depth = np.mean(players[:,0])

        if depth < 0.4:
            return "Defensive"
        elif depth > 0.6:
            return "Attacking"
        else:
            return "Balanced"

    # MAIN DETECTION FUNCTION
    def detect_formation_from_player_positions(self, team_coords, team_side="left"):

        coords = self.normalize_pitch(team_coords)

        if team_side == "right":
            coords[:,0] = 1 - coords[:,0]

        if len(coords) < 10:
            print("⚠️ Warning: Less than 10 player positions detected.")
            return "Unknown", "Unknown", None
        elif len(coords) > 11:
            print("⚠️ Warning: More than 11 player positions detected. Extra players will be ignored for formation detection.")
            coords = coords[:11]
        elif len(coords) == 11:
            print("DEBUG: 11 player positions detected. Removing goalkeeper from analysis for tactical line detection.")

            # remove goalkeeper (deepest player)
            if team_side == "left":
                gk_index = np.argmin(coords[:,0])
            else:
                gk_index = np.argmax(coords[:,0])

            coords = np.delete(coords, gk_index, axis=0)

        print(f"No of coords = {len(coords)}")

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

        for _, row in matches.iterrows():
            diff_score_from_template = self.compare_to_template(coords, row)

            if diff_score_from_template < best_score:
                best_score = diff_score_from_template
                best_matching_formation = row

        mode = self.detect_mode(coords)

        return pattern, mode, best_matching_formation
  
    # Tactical Advice Engine
    def tactical_advice_from_Information_Base(self, formation_row, opponent_row):

        if formation_row is None:
            return [""]

        advice = []

        # Formation description
        desc = formation_row["Description"]
        advice.append(f"Style → {formation_row['Mode']}")

        if isinstance(desc, str):
            advice.append(desc.split(".")[0])

        # Counter formations
        counters = formation_row["Counter Formations"]
        if isinstance(counters, str):
            advice.append(f"Recommended Counters → {counters}")

        # Matchup logic
        if opponent_row is not None:
            if formation_row["Formation Base"] == opponent_row["Formation Base"]:
                advice.append("Mirror matchup → midfield battle expected")

        return advice
    
    def generate_template_from_formation(self, formation, team_side="left", mode="balanced"):
        """
        Creates positions INCLUDING goalkeeper.
        Formation string only represents outfield players.
        Example: 4-3-3 → 10 outfield + 1 GK
        """

        # candidate formations from csv
        matches = self.formation_info_df[
            (self.formation_info_df["Formation"].str.contains(formation, na=False)) &
            (self.formation_info_df["Mode"].str.contains(mode, na=False))
        ]

        if len(matches) == 0:
            return None

        lines = list(map(int, formation.split("-")))
        positions = []

        # 1. ADD GOALKEEPER FIRST
        if team_side == "left":
            gk_x = 3.5                     # inside 6-yard area
        else:
            gk_x = PITCH_LENGTH - 3.5

        gk_y = PITCH_WIDTH / 2
        positions.append([gk_x, gk_y])

        print(f"DEBUG: Added Goalkeeper at position:\n{np.array(positions)}\n")

        template = self.build_template_for_formation_info(matches.iloc[0])

        # 2. OUTFIELD PLAYERS
        if team_side == "left":
            # scale to pitch dimensions
            template[:,0] = template[:,0] * PITCH_LENGTH
            template[:,1] = template[:,1] * PITCH_WIDTH

            positions += template.tolist()
        else:
            # flip horizontally for right side
            flipped_template = template.copy()
            flipped_template[:,0] = 1 - flipped_template[:,0]

            # scale to pitch dimensions
            flipped_template[:,0] = flipped_template[:,0] * PITCH_LENGTH
            flipped_template[:,1] = flipped_template[:,1] * PITCH_WIDTH
            positions += flipped_template.tolist()

        return np.array(positions)