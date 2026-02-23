import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import box
import pandas as pd
from FormationDetector import FormationDetector, get_info_for_formation
from TacticalAnalyzer import TacticalAnalyzer
import Configurations as CF

def voronoi_finite_polygons_2d(vor, radius=1000):
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1]-c[1], vs[:,0]-c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

class InteractiveVoronoiPitch:
    def __init__(self, home_coords, away_coords, home_form_str, away_form_str, home_mode, away_mode):
        
        # Initialize the ML Detector here so we can use it to recalculate on drag
        self.formation_detector = FormationDetector()
        
        # Clean the strings so "Form" doesn't duplicate the "Mode"
        self.home_form_base = home_form_str.replace(home_mode, "").strip()
        self.away_form_base = away_form_str.replace(away_mode, "").strip()
        
        self.home_mode = home_mode
        self.away_mode = away_mode
        
        # Store full strings for database lookups (Counter Formations)
        self.home_form_str = home_form_str
        self.away_form_str = away_form_str

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f"Tactical Board - {CF.MY_TEAM_NAME} vs {CF.OPPONENT_TEAM_NAME}")
        
        self.fig._interactive_board = self 

        self.width = 105
        self.height = 68
        self.bottom_panel_height = 30

        self.tactical_anaylzer = TacticalAnalyzer()
        self.draw_pitch()
        
        self.home_coords = np.array(home_coords, dtype=float)
        self.away_coords = np.array(away_coords, dtype=float)
        self.pitch_poly = box(0, 0, self.width, self.height)
        
        self.voronoi_patches = []
        self.home_texts = []
        self.away_texts = []
        self.home_scatter = None
        self.away_scatter = None
        
        self.selected_point = None
        self.selected_team = None

        self.refresh_graphics()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def draw_pitch(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-self.bottom_panel_height, self.height)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        self.ax.add_patch(Rectangle((0, 0), self.width, self.height, color=CF.PITCH_COLOUR, zorder=0))
        
        line_props = {'linewidth': 2, 'color': 'white', 'zorder': 1}
        self.ax.add_patch(Rectangle((0, 0), self.width, self.height, fill=False, **line_props))
        self.ax.plot([self.width/2, self.width/2], [0, self.height], **line_props)
        self.ax.add_patch(Circle((self.width/2, self.height/2), 9.15, fill=False, **line_props))
        self.ax.add_patch(Rectangle((0, (self.height/2)-20.16), 16.5, 40.32, fill=False, **line_props))
        self.ax.add_patch(Rectangle((self.width-16.5, (self.height/2)-20.16), 16.5, 40.32, fill=False, **line_props))

        self.bottom_panel = Rectangle(
            (0, -self.bottom_panel_height),
            self.width,
            self.bottom_panel_height,
            facecolor="#f4f6f7", edgecolor="black", linewidth=2, zorder=0
        )
        self.ax.add_patch(self.bottom_panel)

        self.left_w = self.width * 0.35
        self.mid_w  = self.width * 0.25
        self.right_w = self.width * 0.40 

        self.ax.plot([self.left_w, self.left_w], [-self.bottom_panel_height, 0], color="black", lw=1.5)
        self.ax.plot([self.left_w + self.mid_w, self.left_w + self.mid_w], [-self.bottom_panel_height, 0], color="black", lw=1.5)

    def compute_voronoi(self):
        try:
            all_points = np.vstack((self.home_coords, self.away_coords))
            epsilon = 1e-6
            all_points = all_points + np.random.uniform(-epsilon, epsilon, all_points.shape)

            vor = Voronoi(all_points)
            regions, vertices = voronoi_finite_polygons_2d(vor)

        except Exception as e:
            print(f"⚠️ Math Error: {e}")
            return [], [], 0, 0

        regions_polys = []
        space_data = []
        h_area = 0
        a_area = 0

        for i, region in enumerate(regions):
            polygon_points = vertices[region]
            poly = ShapelyPoly(polygon_points)
            
            intersection = poly.intersection(self.pitch_poly)
            
            if not intersection.is_empty:
                is_home = i < len(self.home_coords)
                team_str = 'home' if is_home else 'away'
                player_idx = (i + 1) if is_home else (i - len(self.home_coords) + 1)
                
                space_data.append({
                    'id': player_idx,
                    'team': team_str,
                    'area': intersection.area,
                    'centroid': all_points[i]
                })

                geoms = [intersection] if intersection.geom_type == 'Polygon' else intersection.geoms
                for p in geoms:
                    regions_polys.append((p, is_home, player_idx))
                    if is_home: h_area += p.area
                    else: a_area += p.area
                    
        return regions_polys, space_data, h_area, a_area

    def get_goalkeepers(self):
        home_gk = np.argmin(self.home_coords[:,0])
        away_gk = np.argmax(self.away_coords[:,0])
        return home_gk, away_gk

    def refresh_graphics(self):
        if hasattr(self, "bottom_texts"):
            for t in self.bottom_texts:
                t.remove()

        self.bottom_texts = []

        for p in self.voronoi_patches:
            p.remove()
        self.voronoi_patches = []

        # 1. COMPUTE 22-MAN VORONOI (Used for visual rendering & possession control)
        regions, space_data_22, h_area, a_area = self.compute_voronoi()

        # 2. COMPUTE 11-MAN VORONOI (Used exclusively for structural vulnerabilities)
        home_11_space = self.tactical_anaylzer.analyze_space_control(self.home_coords)
        away_11_space = self.tactical_anaylzer.analyze_space_control(self.away_coords)

        # Identify vulnerabilities based purely on isolated team shape
        home_vuln = self.tactical_anaylzer.identify_team_vulnerabilities(home_11_space, 'home', CF.MY_TEAM_NAME)
        away_vuln = self.tactical_anaylzer.identify_team_vulnerabilities(away_11_space, 'away', CF.OPPONENT_TEAM_NAME)
        
        # Combine and sort so biggest gaps appear first
        vulnerabilities = home_vuln + away_vuln
        vulnerabilities.sort(key=lambda v: v['area'], reverse=True)

        # 3. DRAW THE 22-MAN REGIONS WITH DARKER SHADES FOR VULNERABLE PLAYERS
        for poly, is_home, player_idx in regions:
            team_str = 'home' if is_home else 'away'
            color = CF.HOME_PLAYER_COLOUR if is_home else CF.AWAY_PLAYER_COLOUR
            
            # Check if this player was flagged in the 11-man structural calculation
            is_vulnerable = any(
                v['player_id'] == player_idx and v['team_tag'] == team_str 
                for v in vulnerabilities
            )
            
            alpha = 0.85 if is_vulnerable else 0.35 
            
            x, y = poly.exterior.xy
            mpl_poly = Polygon(list(zip(x, y)), facecolor=color, edgecolor='white', alpha=alpha, zorder=2)
            self.ax.add_patch(mpl_poly)
            self.voronoi_patches.append(mpl_poly)

        home_gk, away_gk = self.get_goalkeepers()
        home_colors = [CF.HOME_PLAYER_COLOUR] * len(self.home_coords)
        away_colors = [CF.AWAY_PLAYER_COLOUR] * len(self.away_coords)
        home_sizes = [250] * len(self.home_coords)
        away_sizes = [250] * len(self.away_coords)

        home_colors[home_gk] = CF.HOME_GK_COLOUR
        away_colors[away_gk] = CF.AWAY_GK_COLOUR
        home_sizes[home_gk] = 380
        away_sizes[away_gk] = 380

        if self.home_scatter: self.home_scatter.remove()
        if self.away_scatter: self.away_scatter.remove()

        self.home_scatter = self.ax.scatter(self.home_coords[:,0], self.home_coords[:,1], c=home_colors, s=home_sizes, edgecolors='black', zorder=10)
        self.away_scatter = self.ax.scatter(self.away_coords[:,0], self.away_coords[:,1], c=away_colors, s=away_sizes, edgecolors='black', zorder=10)

        for t in self.home_texts + self.away_texts: t.remove()
        self.home_texts = []
        self.away_texts = []

        for i,(x,y) in enumerate(self.home_coords):
            self.home_texts.append(self.ax.text(x,y,str(i+1), color='white',ha='center',va='center', fontweight='bold',zorder=11))
        for i,(x,y) in enumerate(self.away_coords):
            self.away_texts.append(self.ax.text(x,y,str(i+1), color='white',ha='center',va='center', fontweight='bold',zorder=11))

        if vulnerabilities:
            # Show up to 4 biggest weaknesses
            text_str = "WEAKNESSES:\n" + "\n".join([v['detail'] for v in vulnerabilities[:4]])
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            self.ax.text(0.02, 0.98, text_str, transform=self.ax.transAxes, fontsize=9, verticalalignment='top', color='red', bbox=props, zorder=15)
            
        total = self.width * self.height
        home_control = (h_area / total) * 100 if total else 0
        away_control = (a_area / total) * 100 if total else 0

        home_compact = self.tactical_anaylzer.compactness(self.home_coords)
        away_compact = self.tactical_anaylzer.compactness(self.away_coords)
        home_width = self.tactical_anaylzer.width_usage(self.home_coords)
        away_width = self.tactical_anaylzer.width_usage(self.away_coords)
        center_h, center_a = self.tactical_anaylzer.central_control(self.home_coords, self.away_coords)
        over_h, over_a = self.tactical_anaylzer.overload_score(self.home_coords, self.away_coords)

        compactness = f"{CF.MY_TEAM_NAME} tighter block" if home_compact < away_compact else f"{CF.OPPONENT_TEAM_NAME} tighter block"
        width_analysis = f"{CF.MY_TEAM_NAME} stretching pitch" if home_width > away_width else f"{CF.OPPONENT_TEAM_NAME} stretching pitch"
        central_control = f"{CF.MY_TEAM_NAME} dominance" if center_h > center_a else f"{CF.OPPONENT_TEAM_NAME} dominance"
        overloads = f"{CF.MY_TEAM_NAME} creating overloads" if over_h > over_a else f"{CF.OPPONENT_TEAM_NAME} creating overloads"

        home_row = get_info_for_formation(self.home_form_str)
        away_row = get_info_for_formation(self.away_form_str)
        home_advice, counter_formations = self.tactical_anaylzer.tactical_advice_from_Information_Base(home_row, away_row)

        pad_x = 2
        pad_y = 2
        top_y = -pad_y

        home_display = "Unclear (Low Confidence)" if "Unclear" in self.home_form_base else self.home_form_base
        away_display = "Unclear (Low Confidence)" if "Unclear" in self.away_form_base else self.away_form_base

        home_text = (
            f"[{CF.MY_TEAM_NAME}]\n"  +
            f"Form: {home_display}\n" + 
            f"Mode: {self.home_mode}\n\n" +
            f"[{CF.OPPONENT_TEAM_NAME}]\n" +
            f"Form: {away_display}\n" +
            f"Mode: {self.away_mode}"
        )
        self.bottom_texts.append(self.ax.text(pad_x, top_y, home_text, ha="left", va="top", fontsize=10, linespacing=1.5))

        counter_formations = "\n".join(counter_formations)
        control_text = (
            "Space Control\n"
            f"{CF.MY_TEAM_NAME}: {home_control:.1f}%\n"
            f"{CF.OPPONENT_TEAM_NAME}: {away_control:.1f}%\n\n"
        )
        if counter_formations:
            control_text += f"Counters:\n{counter_formations}"
        else:
            control_text += "Counters:\nN/A"

        self.bottom_texts.append(self.ax.text(self.left_w + pad_x, top_y, control_text, ha="left", va="top", fontsize=10, linespacing=1.5))

        structure_text = (
            f"Compactness:\n{compactness}\n"
            f"Width:\n{width_analysis}\n"
            f"Central:\n{central_control}\n"
            f"Overloads:\n{overloads}"
        )

        self.bottom_texts.append(self.ax.text(self.left_w + self.mid_w + pad_x, top_y, structure_text, ha="left", va="top", fontsize=10, linespacing=1.5, wrap=True))

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return 
        
        click = np.array([event.xdata, event.ydata])
        
        dist_h = np.linalg.norm(self.home_coords - click, axis=1)
        if np.min(dist_h) < 5.0:
            self.selected_point = np.argmin(dist_h)
            self.selected_team = 'home'
            return

        dist_a = np.linalg.norm(self.away_coords - click, axis=1)
        if np.min(dist_a) < 5.0:
            self.selected_point = np.argmin(dist_a)
            self.selected_team = 'away'

    def on_motion(self, event):
        if self.selected_point is None or event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return 
        
        x, y = np.clip(event.xdata, 0, self.width), np.clip(event.ydata, 0, self.height)
        
        if self.selected_team == 'home':
            self.home_coords[self.selected_point] = [x, y]
            self.home_scatter.set_offsets(self.home_coords)
            self.home_texts[self.selected_point].set_position((x, y))
        else:
            self.away_coords[self.selected_point] = [x, y]
            self.away_scatter.set_offsets(self.away_coords)
            self.away_texts[self.selected_point].set_position((x, y))
            
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.selected_point is not None:
            self.selected_point = None
            self.selected_team = None
            
            # Recalculate formations based on newly dragged positions
            _, new_home_mode, new_home_str = self.formation_detector.detect_formation_from_player_positions(self.home_coords, team_side="left")
            self.home_form_base = new_home_str.replace(new_home_mode, "").strip()
            self.home_mode = new_home_mode
            self.home_form_str = new_home_str

            _, new_away_mode, new_away_str = self.formation_detector.detect_formation_from_player_positions(self.away_coords, team_side="right")
            self.away_form_base = new_away_str.replace(new_away_mode, "").strip()
            self.away_mode = new_away_mode
            self.away_form_str = new_away_str
            
            self.refresh_graphics()