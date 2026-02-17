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
        self.role_columns = CF.ROLE_LIST

        self.role_columns = [
            c for c in self.role_columns
            if c in self.formation_info_df.columns
            and c != CF.ROLE_Goalkeeper
        ]

        # 5x5 Tactical Grid (normalized)
        # Keep everyone before penalty area
        self.depth_grid = np.array([
            0.20,  # DEF
            0.35,  # LOW MID
            0.50,  # MID
            0.65,  # HIGH MID
            0.80   # ATT (before penalty box)
        ])

        self.width_grid = np.array([
            0.15,
            0.325,
            0.50,
            0.675,
            0.85
        ])

        # Role → (depth_index, width_index)
        self.role_zone_map = {
            # DEF LINE
            CF.ROLE_Left_Back: (0, 0),
            CF.ROLE_Left_Center_Back: (0, 1),
            CF.ROLE_Center_Back: (0, 2),
            CF.ROLE_Right_Center_Back: (0, 3),
            CF.ROLE_Right_Back: (0, 4),

            # DEF MID
            CF.ROLE_Left_Defensive_Midfielder: (1, 1),
            CF.ROLE_Central_Defensive_Midfielder: (1, 2),
            CF.ROLE_Right_Defensive_Midfielder: (1, 3),

            # MID
            CF.ROLE_Left_Midfielder: (2, 0),
            CF.ROLE_Left_Central_Midfielder : (2, 1),
            CF.ROLE_Central_Midfielder : (2, 2),
            CF.ROLE_Right_Central_Midfielder: (2, 3),
            CF.ROLE_Right_Midfielder: (2, 4),

            # ATT MID
            CF.ROLE_Left_Attacking_Midfielder: (3, 1),
            CF.ROLE_Central_Attacking_Midfielder: (3, 2),
            CF.ROLE_Right_Attacking_Midfielder: (3, 3),

            # WINGS
            CF.ROLE_Left_Winger: (4, 0),
            CF.ROLE_Left_Forward: (4, 1),
            CF.ROLE_Center_Forward: (4, 2),
            CF.ROLE_Right_Forward: (4, 3),
            CF.ROLE_Right_Winger: (4, 4),

            # STRIKERS
            CF.ROLE_Left_Striker: (4, 1),
            CF.ROLE_Striker: (4, 2),
            CF.ROLE_Right_Striker: (4, 3),

            # Wingbacks
            CF.ROLE_Left_Wing_Back: (1, 0),
            CF.ROLE_Right_Wing_Back: (1, 4),
        }

    def extract_structure(self, formation_name):
        try:
            return [int(x) for x in formation_name.split("-")]
        except:
            return []

    def build_template_from_csv_row(self, formation_row, mode=CF.MODE_BALANCED):

        template = []

        # Mode depth shifting
        mode_shift = {
            CF.MODE_DEFENSIVE: -0.08,
            CF.MODE_BALANCED: 0.0,
            CF.MODE_ATTACKING: 0.08
        }

        depth_adjust = mode_shift.get(mode, 0.0)

        # Group roles by depth line

        line_groups = {}

        for role in self.role_columns:

            if role not in formation_row:
                continue

            count = int(formation_row[role])
            if count <= 0:
                continue

            if role not in self.role_zone_map:
                continue

            depth_i, _ = self.role_zone_map[role]

            if depth_i not in line_groups:
                line_groups[depth_i] = []

            line_groups[depth_i].append((role, count))

        # Build each tactical line dynamically

        for depth_i in sorted(line_groups.keys()):

            roles = line_groups[depth_i]

            total_players_line = sum(count for _, count in roles)

            base_x = self.depth_grid[depth_i] + depth_adjust

            # Dynamically center players across pitch width
            width_positions = np.linspace(0.15, 0.85, total_players_line)

            idx = 0

            for role, count in roles:
                for _ in range(count):

                    y = width_positions[idx]

                    template.append([base_x, y])
                    idx += 1

        return np.array(template, dtype=float)

    # Compare real players to formation template
    def compare_to_template(self, players, formation_row):

        template = self.build_template_from_csv_row(formation_row)

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
        mode=CF.MODE_BALANCED
    ):

        matches = self.formation_info_df[
            self.formation_info_df["Formation"] == formation_name
        ]

        if len(matches) == 0:
            raise ValueError(f"Formation '{formation_name}' not found in CSV")

        formation_row = matches.iloc[0]

        template = self.build_template_from_csv_row(
            formation_row,
            mode
        )

        # Clip bounds
        template[:,0] = np.clip(template[:,0], 0.05, 0.95)
        template[:,1] = np.clip(template[:,1], 0.05, 0.95)

        # Flip side
        if team_side == "right":
            template[:,0] = 1 - template[:,0]

        # Scale to pitch
        template[:,0] *= CF.PITCH_LENGTH
        template[:,1] *= CF.PITCH_WIDTH

        # Add GK
        if team_side == "left":
            gk_x = 0.035
        else:
            gk_x = 0.965

        gk = np.array([[gk_x * CF.PITCH_LENGTH,
                        0.5 * CF.PITCH_WIDTH]])

        template = np.vstack((gk, template))

        return template