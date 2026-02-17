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
                "‼️ FormationDetector expects CSV path or pandas DataFrame"
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
            role
            for role in CF.Role
            if role != CF.Role.Goalkeeper
            and role.value in self.formation_info_df.columns
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
            CF.Role.Left_Back: (0, 0),
            CF.Role.Left_Center_Back: (0, 1),
            CF.Role.Center_Back: (0, 2),
            CF.Role.Right_Center_Back: (0, 3),
            CF.Role.Right_Back: (0, 4),

            # DEF MID
            CF.Role.Left_Defensive_Midfielder: (1, 1),
            CF.Role.Central_Defensive_Midfielder: (1, 2),
            CF.Role.Right_Defensive_Midfielder: (1, 3),

            # MID
            CF.Role.Left_Midfielder: (2, 0),
            CF.Role.Left_Central_Midfielder : (2, 1),
            CF.Role.Central_Midfielder : (2, 2),
            CF.Role.Right_Central_Midfielder: (2, 3),
            CF.Role.Right_Midfielder: (2, 4),

            # ATT MID
            CF.Role.Left_Attacking_Midfielder: (3, 1),
            CF.Role.Central_Attacking_Midfielder: (3, 2),
            CF.Role.Right_Attacking_Midfielder: (3, 3),

            # WINGS
            CF.Role.Left_Winger: (4, 0),
            CF.Role.Left_Forward: (4, 1),
            CF.Role.Center_Forward: (4, 2),
            CF.Role.Right_Forward: (4, 3),
            CF.Role.Right_Winger: (4, 4),

            # STRIKERS
            CF.Role.Left_Striker: (4, 1),
            CF.Role.Striker: (4, 2),
            CF.Role.Right_Striker: (4, 3),

            # Wingbacks
            CF.Role.Left_Wing_Back: (1, 0),
            CF.Role.Right_Wing_Back: (1, 4),
        }

    def extract_structure(self, formation_name):
        try:
            return [int(x) for x in formation_name.split("-")]
        except:
            return []

    def build_template_from_csv_row(self, formation_row, mode=CF.Mode.BALANCED):
        if formation_row is None:
            raise ValueError(f"‼️ [ERROR] Formation row is empty!")

        template = []

        # Mode depth shifting
        mode_shift = {
            CF.Mode.DEFENSIVE: -0.085,
            CF.Mode.DEFENDING: -0.085,
            CF.Mode.BALANCED: 0.0,
            CF.Mode.ATTACKING: 0.1
        }

        depth_adjust = mode_shift.get(mode, 0.0)

        # Group roles by depth line

        line_groups = {}

        for role in self.role_columns:
            column_name = role.value

            if column_name not in formation_row:
                raise ValueError(
                    f"‼️ Role-{column_name} not found in formation row"
                )

            count = int(formation_row[column_name])
            if count <= 0:
                continue

            if role not in self.role_zone_map:
                raise ValueError(
                    f"‼️ Role-{role} not found in role_zone_map"
                )

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
        formation_struct,
        mode = CF.Mode.BALANCED,
        shape = CF.Shape.NA,
        team_side="left"
    ):
        form_info_df = self.formation_info_df.copy()

        # If formation string contains shape (e.g. "4-3-3 Diamond")
        if isinstance(formation_struct, str) and " " in formation_struct:
            parts = formation_struct.split(" ", 1)  # split only once
            formation_struct = parts[0]
            shape = parts[1]
        
        # Normalize strings for safety
        formation_struct = str(formation_struct).strip()
        mode = str(mode).strip()
        shape = str(shape).strip()

        # Try Exact Match
        matches = form_info_df[
            (form_info_df["Structure"] == formation_struct) &
            (form_info_df["Mode"] == mode) &
            (form_info_df["Shape"] == shape)
        ]

        # Ignore Mode
        if len(matches) == 0:
            matches = form_info_df[
                (form_info_df["Structure"] == formation_struct) &
                (form_info_df["Shape"] == shape)
            ]

        # Ignore Shape
        if len(matches) == 0:
            matches = form_info_df[
                (form_info_df["Structure"] == formation_struct) &
                (form_info_df["Mode"] == mode)
            ]

        # Ignore Mode and Shape, use basic structure
        if len(matches) == 0:
            matches = form_info_df[
                form_info_df["Structure"] == formation_struct
            ]

        # If no match found, fallback to Balanced version
        if len(matches) == 0:
            print(f"⚠️ Warning: Formation '{formation_struct}' not found in DB. Falling back to 4-4-2 Midfield.")
            
            matches = form_info_df[
                (form_info_df["Structure"] == "4-4-2") &
                (form_info_df["Mode"] == CF.Mode.MIDFIELD)
            ]

        formation_row = matches.iloc[0]

        template = self.build_template_from_csv_row(
            formation_row,
            mode
        )

        if len(template) == 0:
            raise ValueError(f"‼️ Template empty!")

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