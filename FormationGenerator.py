import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import Configurations as CF

class FormationGenerator:
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

        self.depth_levels = {
            "DEF": 0.22,
            "LOW_MID": 0.38,
            "MID": 0.52,
            "HIGH_MID": 0.68,
            "ATT": 0.85
        }

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

        # lateral meaning of role (LEFT / CENTER / RIGHT)
        self.role_lateral_bias = {
            "LB": -1, "LCB": -0.5, "CB": 0,
            "RCB": 0.5, "RB": 1,

            "LM": -1, "LCM": -0.4, "CM": 0,
            "RCM": 0.4, "RM": 1,

            "LW": -1, "RW": 1,

            "LAM": -0.5, "CAM": 0, "RAM": 0.5,

            "LS": -0.4, "ST": 0, "RS": 0.4,
            "CF": 0,

            "LWB": -1, "RWB": 1,
        }

    def extract_structure(self, formation_name):
        try:
            return [int(x) for x in formation_name.split("-")]
        except:
            return []

    # Create template positions from CSV row
    def build_template_for_formation_info(self, formation_row):

        formation_name = formation_row["Formation"]
        structure = self.extract_structure(formation_name)

        if len(structure) == 0:
            return np.array([])

        # defensive → attacking depths from structure
        x_levels = np.linspace(0.25, 0.85, len(structure))

        # -------- collect players by line ----------
        lines = [[] for _ in structure]

        role_counts = {}

        for role in self.role_columns:

            if role not in formation_row:
                continue

            count = int(formation_row[role])
            if count == 0:
                continue

            role_counts[role] = count

        # assign roles sequentially to structure lines
        role_list = []
        for role, count in role_counts.items():
            role_list += [role] * count

        idx = 0
        for line_i, line_size in enumerate(structure):
            lines[line_i] = role_list[idx:idx+line_size]
            idx += line_size

        template = []

        # -------- build coordinates ----------
        for line_i, roles in enumerate(lines):

            x = x_levels[line_i]

            if len(roles) == 0:
                continue

            # structural spacing
            base_y = np.linspace(0.1, 0.9, len(roles))

            for i, role in enumerate(roles):

                y = base_y[i]

                # apply semantic role bias
                bias = self.role_lateral_bias.get(role, 0)

                y += bias * 0.08   # strength of role positioning

                y = np.clip(y, 0.05, 0.95)

                template.append([x, y])

        return np.array(template, dtype=float)

    def build_template_from_formation_name(self, formation_name):
        structure = self.extract_structure(formation_name)

        if len(structure) == 0:
            return np.array([])

        # defensive → attacking spacing
        x_levels = np.linspace(0.25, 0.85, len(structure))

        template = []

        for line_i, players_in_line in enumerate(structure):

            x = x_levels[line_i]

            # evenly distribute across pitch width
            y_positions = np.linspace(0.1, 0.9, players_in_line)

            for y in y_positions:
                template.append([x, y])

        return np.array(template, dtype=float)

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
    
    def generate_template_from_formation(
        self,
        formation_name,
        team_side="left",
        mode="balanced"
    ):

        matches = self.formation_info_df[
            self.formation_info_df["Formation"] == formation_name
        ]

        print(f"Matching formation '{formation_name}': ")
        print(matches)

        if len(matches) == 0:
            # fallback structural generation
            template = self.build_template_from_formation_name(formation_name)
        else:
            formation_row = matches.iloc[0]
            print("Using formation info from CSV:")
            print(formation_row)
            template = self.build_template_for_formation_info(formation_row)

        print("Template:")
        print(template)

        # -------- MODE ADJUSTMENTS ----------
        if CF.MODE_DEFENSIVE in mode == CF.MODE_DEFENSIVE:
            template[:,0] -= 0.05
            template[:,1] = 0.5 + (template[:,1]-0.5)*0.75

        elif CF.MODE_ATTACKING in mode:
            template[:,0] += 0.05
            template[:,1] = 0.5 + (template[:,1]-0.5)*1.15

        # -------- flip side ----------
        if team_side == "right":
            template[:,0] = 1 - template[:,0]

        # -------- ADD GOALKEEPER (normalized coordinates) ----------
        if team_side == "left":
            gk_x = 0.035          # ≈ 3.5m / 105m
        else:
            gk_x = 0.965          # symmetric position

        gk_y = 0.5                # center width

        template = np.vstack(([gk_x, gk_y], template))

        # -------- scale to pitch ----------
        template[:,0] *= CF.PITCH_LENGTH
        template[:,1] *= CF.PITCH_WIDTH

        return template           