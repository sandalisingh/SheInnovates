import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

FORMATIONS_INFO_DB = pd.read_csv("Data/Formations_info.csv")

PITCH_LENGTH = 105
PITCH_WIDTH = 68
PITCH_COLOUR = '#2e8b57'

AREA_CAPTURE_MIN_THRESHOLD = 550 # in m^2

class TacticalAnalyzer:
    def __init__(self):
        self.pitch_box = box(0, 0, PITCH_LENGTH, PITCH_WIDTH)

    def analyze_space_control(self, centroids):
        # Voronoi with Boundary Points
        boundary_points = [
            [-20, -20], 
            [PITCH_LENGTH + 20, -20], 
            [PITCH_LENGTH + 20, PITCH_WIDTH + 20], 
            [-20, PITCH_WIDTH + 20]
        ]
        points = np.vstack([centroids, boundary_points])
        
        voronoi = Voronoi(points)
        analysis_results = []
        
        for i in range(len(centroids)):
            region_idx = voronoi.point_region[i]
            region_verts = voronoi.regions[region_idx]
            
            if -1 in region_verts or len(region_verts) == 0:
                continue
                
            poly = Polygon(voronoi.vertices[region_verts])
            # Clip to Pitch Box
            intersection = poly.intersection(self.pitch_box)
            
            analysis_results.append({
                'id': i,
                'area': intersection.area,
                'poly': intersection,
                'centroid': centroids[i]
            })
            
        return analysis_results

    def identify_vulnerabilities(self, space_data):
        vulnerabilities = []
        
        for p in space_data:
            x, y = p['centroid']
            area = p['area']
            
            # Check central band (30m - 75m length)
            if 30 < x < 75: # player is in central zone
                if area > AREA_CAPTURE_MIN_THRESHOLD:
                    vulnerabilities.append({
                        'type': 'Sparse Coverage',
                        'detail': f"ID {p['id']} covers {int(area)}m²",
                        'player_data': p
                    })
        return vulnerabilities

    def visualize_using_voronoi(self, centroids, formation_name, space_data, vulnerabilities):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.set_facecolor(PITCH_COLOUR) # Pitch Green
        ax.set_title(f"Tactical Analysis for Formation : {formation_name}", fontsize=14, color='white', pad=20)
        
        # Draw Pitch Outline (0 to 105, 0 to 68)
        ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color='white', linewidth=2)
        ax.plot([52.5, 52.5], [0, 68], 'w--') # Halfway line

        # Draw Voronoi Regions
        for p in space_data:
            is_vulnerable = any(v['player_data']['id'] == p['id'] for v in vulnerabilities)
            color = '#ff4d4d' if is_vulnerable else '#1a75ff'
            alpha = 0.6 if is_vulnerable else 0.2
            
            if not p['poly'].is_empty:
                x, y = p['poly'].exterior.xy
                ax.fill(x, y, fc=color, ec='white', alpha=alpha, linewidth=1)
                ax.text(p['centroid'][0], p['centroid'][1]-2, f"{int(p['area'])}", 
                        color='white', fontsize=8, ha='center')

        # Draw Players
        ax.scatter(centroids[:, 0], centroids[:, 1], c='white', s=100, edgecolors='black', zorder=5)
        
        # Annotate Vulnerabilities
        if vulnerabilities:
            text_str = "WEAKNESSES:\n" + "\n".join([v['detail'] for v in vulnerabilities[:3]])
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', color='red', bbox=props)

        # Force limits to show "out of bounds" if scaling failed, 
        # but keep aspect ratio roughly correct
        ax.set_xlim(-5, 110)
        ax.set_ylim(-5, 75)
        
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        plt.show()

    def compactness(self, team):
        centroid = np.mean(team, axis=0)
        distances = np.linalg.norm(team - centroid, axis=1)
        return np.mean(distances)
    
    def width_usage(self, team):
        return np.max(team[:,1]) - np.min(team[:,1])
    
    def vertical_structure(self, team):
        xs = np.sort(team[:,0])
        thirds = np.array_split(xs, 3)

        line_means = [np.mean(t) for t in thirds]
        return np.diff(line_means)  # defense-mid, mid-attack

    def central_control(self, home, away):
        center_box = np.array([
            [PITCH_WIDTH*0.35, PITCH_LENGTH*0.25],
            [PITCH_WIDTH*0.65, PITCH_LENGTH*0.75]
        ])

        def inside(team):
            return np.sum(
                (team[:,0] > center_box[0,0]) &
                (team[:,0] < center_box[1,0]) &
                (team[:,1] > center_box[0,1]) &
                (team[:,1] < center_box[1,1])
            )

        return inside(home), inside(away)

    def overload_score(self, home, away, radius=12):
        score_home = 0
        score_away = 0

        for h in home:
            h_near = np.sum(np.linalg.norm(home-h,axis=1)<radius)
            a_near = np.sum(np.linalg.norm(away-h,axis=1)<radius)
            if h_near > a_near:
                score_home += 1

        for a in away:
            a_near = np.sum(np.linalg.norm(away-a,axis=1)<radius)
            h_near = np.sum(np.linalg.norm(home-a,axis=1)<radius)
            if a_near > h_near:
                score_away += 1

        return score_home, score_away

    def tactical_advice_from_Information_Base(self, my_team_row, opponent_row):
        advice = []
        counter_formations = []

        # if my_team_row is not None:
        #     advice.append(f"{my_team_row['Formation']} → {my_team_row['Description']}")

        # Matchup logic
        if opponent_row is not None:
            counter_str = opponent_row.get("Counter Formations", "")

            if isinstance(counter_str, str) and counter_str.strip():
                counters = [c.strip() for c in counter_str.split("/")]
                counter_formations.extend(counters)   # ← FIX: use extend instead of append

        return advice, counter_formations

