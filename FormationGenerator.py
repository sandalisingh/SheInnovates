import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import Configurations as CF

class FormationGenerator:
    def __init__(self, formation_info_df=CF.FORMATIONS_INFO_DB):
        if isinstance(formation_info_df, pd.DataFrame):
            self.formation_info_df = formation_info_df.copy()
        elif isinstance(formation_info_df, str):
            self.formation_info_df = pd.read_csv(formation_info_df)
        else:
            raise ValueError("FormationGenerator expects CSV path or pandas DataFrame")

        self.role_columns = [r for r in CF.Role if r != CF.Role.Goalkeeper]

        # FIXED: 6-Tier Depth Grid (Separates CAM, CF, and ST perfectly)
        self.depth_grid = np.array([
            0.18,  # 0: DEF
            0.30,  # 1: DEF MID 
            0.45,  # 2: MID 
            0.60,  # 3: ATT MID 
            0.72,  # 4: FORWARDS / WINGS (CF)
            0.82   # 5: STRIKERS (ST)
        ])
        
        self.width_grid = np.array([0.15, 0.325, 0.50, 0.675, 0.85])

        self.role_zone_map = {
            CF.Role.Left_Back: (0, 0),
            CF.Role.Left_Center_Back: (0, 1),
            CF.Role.Center_Back: (0, 2),
            CF.Role.Right_Center_Back: (0, 3),
            CF.Role.Right_Back: (0, 4),

            CF.Role.Left_Wing_Back: (1, 0),
            CF.Role.Left_Defensive_Midfielder: (1, 1),
            CF.Role.Central_Defensive_Midfielder: (1, 2),
            CF.Role.Right_Defensive_Midfielder: (1, 3),
            CF.Role.Right_Wing_Back: (1, 4),

            CF.Role.Left_Midfielder: (2, 0),
            CF.Role.Left_Central_Midfielder : (2, 1),
            CF.Role.Central_Midfielder : (2, 2),
            CF.Role.Right_Central_Midfielder: (2, 3),
            CF.Role.Right_Midfielder: (2, 4),

            CF.Role.Left_Attacking_Midfielder: (3, 1),
            CF.Role.Central_Attacking_Midfielder: (3, 2),
            CF.Role.Right_Attacking_Midfielder: (3, 3),

            CF.Role.Left_Winger: (4, 0),
            CF.Role.Left_Forward: (4, 1),
            CF.Role.Center_Forward: (4, 2),
            CF.Role.Right_Forward: (4, 3),
            CF.Role.Right_Winger: (4, 4),
            
            # Strikers push to the exclusive 6th tier
            CF.Role.Left_Striker: (5, 1),
            CF.Role.Striker: (5, 2),
            CF.Role.Right_Striker: (5, 3),
        }

    def build_template_from_roles(self, formation_row, mode=CF.Mode.BALANCED.value):
        if formation_row is None:
            raise ValueError(f"‼️ [ERROR] Formation row is empty!")

        template = []
        mode_key = mode.value if hasattr(mode, 'value') else str(mode)

        mode_shift = {
            CF.Mode.DEFENSIVE.value: -0.085,
            CF.Mode.DEFENDING.value: -0.085,
            CF.Mode.BALANCED.value: 0.0,
            CF.Mode.MIDFIELD.value: 0.0,
            CF.Mode.HOLDING.value: -0.05,
            CF.Mode.ATTACKING.value: 0.1
        }

        depth_adjust = mode_shift.get(mode_key, 0.0)
        line_groups = {}

        for role in self.role_columns:
            column_name = role.value
            if column_name not in formation_row:
                continue

            count = int(formation_row[column_name])
            if count <= 0 or role not in self.role_zone_map:
                continue

            depth_i, width_i = self.role_zone_map[role]
            if depth_i not in line_groups:
                line_groups[depth_i] = []

            line_groups[depth_i].append((role, count, width_i))

        for depth_i in sorted(line_groups.keys()):
            roles = line_groups[depth_i]
            base_x = self.depth_grid[depth_i] + depth_adjust

            roles.sort(key=lambda item: item[2])

            for role, count, width_i in roles:
                for j in range(count):
                    offset = (j - (count-1)/2.0) * 0.05 if count > 1 else 0
                    y = self.width_grid[width_i] + offset
                    template.append([base_x, y])

        return np.array(template, dtype=float)

    def compare_to_template(self, players, formation_row):
        template = self.build_template_from_roles(formation_row)
        if len(template) == 0:
            return 1e9

        players = np.array(players)
        if players.shape[0] > 0 and players[:, 0].max() > 2.0:
            players_norm = players.copy()
            players_norm[:, 0] /= CF.PITCH_LENGTH
            players_norm[:, 1] /= CF.PITCH_WIDTH
            players = players_norm

        n = min(len(players), len(template))
        players = players[:n]
        template = template[:n]

        cost = cdist(players, template)
        row_ind, col_ind = linear_sum_assignment(cost)
        return cost[row_ind, col_ind].mean()
    
    def extract_structure(self, formation_name):
        try:
            structure_str = str(formation_name).split(" ")[0]
            return [int(x) for x in structure_str.split("-")]
        except Exception as e:
            return []
    
    def build_template_from_formation_name(self, formation_name):
        structure = self.extract_structure(formation_name)
        if len(structure) == 0:
            return np.array([])

        x_levels = np.linspace(0.25, 0.82, len(structure))
        template = []

        for line_i, players_in_line in enumerate(structure):
            x = x_levels[line_i]
            if players_in_line == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.15, 0.85, players_in_line)

            for y in y_positions:
                template.append([x, y])

        return np.array(template, dtype=float)
    
    def generate_template_from_formation(self, formation_name, team_side="left", mode=CF.Mode.BALANCED.value):
        from FormationDetector import get_info_for_formation
        
        formation_row = get_info_for_formation(formation_name)

        # print("Detected formation:")
        # print(formation_row)
        
        if formation_row is not None:
            template = self.build_template_from_roles(formation_row, mode)
        else:
            template = self.build_template_from_formation_name(formation_name)

        if len(template) == 0:
            return np.array([[]]) 

        mode_str = mode.value if hasattr(mode, 'value') else str(mode)

        if CF.Mode.DEFENSIVE.value in mode_str or CF.Mode.DEFENDING.value in mode_str:
            template[:,0] -= 0.05
            template[:,1] = 0.5 + (template[:,1]-0.5)*0.75
        elif CF.Mode.ATTACKING.value in mode_str:
            template[:,0] += 0.05
            template[:,1] = 0.5 + (template[:,1]-0.5)*1.15

        # Restrict all outfield players to stay outside the 16.5m penalty box
        # 16.5m / 105m = ~0.157 boundary on both ends
        template[:,0] = np.clip(template[:,0], 0.16, 0.84)

        if team_side == "right":
            template[:,0] = 1 - template[:,0]

        gk_x = 0.035 if team_side == "left" else 0.965

        if template.ndim == 1:
            template = template.reshape(-1, 2)
            
        template = np.vstack(([gk_x, 0.5], template))

        template[:,0] *= CF.PITCH_LENGTH
        template[:,1] *= CF.PITCH_WIDTH

        return template