import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import Configurations as CF
from FormationDetector import get_row_from_string

FORMATIONS_INFO_DB = pd.read_csv("Data/Formations_info.csv")
PITCH_LENGTH = 105
PITCH_WIDTH = 68
PITCH_COLOUR = '#2e8b57'
AREA_CAPTURE_MIN_THRESHOLD = 550 # in m^2

class TacticalAnalyzer:
    def __init__(self):
        self.pitch_box = box(0, 0, PITCH_LENGTH, PITCH_WIDTH)

    # --- For Voronoi area calculation ---
    def analyze_space_control(self, centroids):
        # Voronoi with Boundary Points to prevent infinite regions
        boundary_points = [
            [-20, -20], 
            [PITCH_LENGTH + 20, -20], 
            [PITCH_LENGTH + 20, PITCH_WIDTH + 20], 
            [-20, PITCH_WIDTH + 20]
        ]
        points = np.vstack([centroids, boundary_points])
        
        # Add jitter to prevent degenerate geometry crashes
        epsilon = 1e-6
        points = points + np.random.uniform(-epsilon, epsilon, points.shape)
        
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
    # ---------------------------------------------------

    def identify_vulnerabilities(self, space_data):
        vulnerabilities = []
        
        for p in space_data:
            x, y = p['centroid']
            area = p['area']
            
            # Check central band (30m - 75m length)
            if 30 < x < 75: 
                if area > AREA_CAPTURE_MIN_THRESHOLD:
                    # Determine team name based on the tag we injected in ImageAnalyzer/VideoAnalyzer
                    team_prefix = CF.MY_TEAM_NAME if p.get('team') == 'home' else CF.OPPONENT_TEAM_NAME
                    
                    vulnerabilities.append({
                        'type': 'Sparse Coverage',
                        'detail': f"{team_prefix} Player {p['id']} gap: {int(area)}m²",
                        'player_data': p
                    })
                    
        # Sort so the largest, most dangerous areas appear at the top of the list
        vulnerabilities.sort(key=lambda v: v['player_data']['area'], reverse=True)
        return vulnerabilities

    def compactness(self, team):
        if len(team) == 0: return 0
        centroid = np.mean(team, axis=0)
        distances = np.linalg.norm(team - centroid, axis=1)
        return np.mean(distances)
    
    def width_usage(self, team):
        if len(team) == 0: return 0
        return np.max(team[:,1]) - np.min(team[:,1])
    
    def vertical_structure(self, team):
        if len(team) == 0: return np.array([])
        xs = np.sort(team[:,0])
        thirds = np.array_split(xs, 3)

        line_means = [np.mean(t) if len(t) > 0 else 0 for t in thirds]
        return np.diff(line_means)  

    def central_control(self, home, away):
        center_box = np.array([
            [PITCH_WIDTH*0.35, PITCH_LENGTH*0.25],
            [PITCH_WIDTH*0.65, PITCH_LENGTH*0.75]
        ])

        def inside(team):
            if len(team) == 0: return 0
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
        
        if len(home) == 0 or len(away) == 0: return 0, 0

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
        global FORMATIONS_INFO_DB
        
        my_team_full_row = get_row_from_string(my_team_row)
        opponent_full_row = get_row_from_string(opponent_row)

        if my_team_full_row is not None and isinstance(my_team_full_row, pd.Series):
             desc = my_team_full_row.get("Description", "")
             if isinstance(desc, str) and desc.strip():
                 advice.append(f"{my_team_full_row.get('Formation', 'Formation')} → {desc}")

        if opponent_full_row is not None and isinstance(opponent_full_row, pd.Series):
            counter_str = opponent_full_row.get("Counter Formations", "")
            if isinstance(counter_str, str) and counter_str.strip():
                counters = [c.strip() for c in counter_str.split("/")]
                counter_formations.extend(counters)   

        return advice, counter_formations