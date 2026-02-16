import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import box
import pandas as pd
from FormationDetector import FormationDetector
from TacticalAnalyzer import TacticalAnalyzer
import textwrap

# ----------------------------- CONFIGURATION -----------------------------

FORMATIONS_INFO_DB = pd.read_csv("Data/Formations_info.csv")

PITCH_LENGTH = 105
PITCH_WIDTH = 68
MY_TEAM_NAME = "Home"
OPPONENT_TEAM_NAME = "Away"

HOME_PLAYER_COLOUR = '#3498db'
HOME_GK_COLOUR = "#0ff1b5"

AWAY_PLAYER_COLOUR = '#e74c3c'
AWAY_GK_COLOUR = '#e67e22'

# ------------------------- VORONOI GENERATION -------------------------
def voronoi_finite_polygons_2d(vor, radius=1000):
    """
    Reconstruct infinite Voronoi regions to finite regions.
    Source adapted from SciPy documentation.
    """
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
    def __init__(self, home_formation, away_formation):
        self.formation_detector = FormationDetector(FORMATIONS_INFO_DB)

        # 1. Setup the Figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f"Tactical Board - {MY_TEAM_NAME} vs {OPPONENT_TEAM_NAME}")
        
        # Dimensions
        self.width = 105
        self.height = 68
        self.bottom_panel_height = 25

        self.analyzer = TacticalAnalyzer()
        
        # 2. Draw the static pitch IMMEDIATELY (so you don't see a blank screen)
        self.draw_pitch()
        
        # 3. Setup Data
        self.home_coords = np.array(home_formation, dtype=float)
        self.away_coords = np.array(away_formation, dtype=float)
        self.pitch_poly = box(0, 0, self.width, self.height)
        
        # Graphics Containers
        self.voronoi_patches = []
        self.home_texts = []
        self.away_texts = []
        self.home_scatter = None
        self.away_scatter = None
        
        # State
        self.selected_point = None
        self.selected_team = None

        # 4. Initial Calculation & Draw

        # Draw players once
        self.refresh_graphics()
        
        # 5. Connect Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def draw_pitch(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-self.bottom_panel_height, self.height)

        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Green background
        self.ax.add_patch(Rectangle((0, 0), self.width, self.height, color='#2ecc71', zorder=0))
        
        # White lines
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
            facecolor="#f4f6f7",
            edgecolor="black",
            linewidth=2,
            zorder=0
        )

        self.ax.add_patch(self.bottom_panel)

        self.left_w = self.width * 0.35
        self.mid_w  = self.width * 0.2
        self.right_w = self.width * 0.45

        # vertical separators
        self.ax.plot([self.left_w, self.left_w],
                    [-self.bottom_panel_height, 0],
                    color="black", lw=1.5)

        self.ax.plot([self.left_w + self.mid_w,
                    self.left_w + self.mid_w],
                    [-self.bottom_panel_height, 0],
                    color="black", lw=1.5)

    def compute_voronoi(self):
        # Robust Voronoi calculation
        try:
            all_points = np.vstack((self.home_coords, self.away_coords))
            vor = Voronoi(all_points)

            regions, vertices = voronoi_finite_polygons_2d(vor)

        except Exception as e:
            print(f"Math Error: {e}")
            return [], 0, 0

        regions_polys = []
        h_area = 0
        a_area = 0

        for i, region in enumerate(regions):
            polygon_points = vertices[region]
            poly = ShapelyPoly(polygon_points)
            
            # Clip to pitch
            intersection = poly.intersection(self.pitch_poly)
            
            if not intersection.is_empty:
                geoms = [intersection] if intersection.geom_type == 'Polygon' else intersection.geoms
                for p in geoms:
                    is_home = i < len(self.home_coords)
                    regions_polys.append((p, is_home))
                    if is_home: h_area += p.area
                    else: a_area += p.area
                    
        return regions_polys, h_area, a_area
    
    def get_goalkeepers(self):
        # Home attacks right → GK is smallest X
        home_gk = np.argmin(self.home_coords[:,0])

        # Away attacks left → GK is largest X
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

        # COMPUTE VORONOI
        regions, h_area, a_area = self.compute_voronoi()

        # DRAW VORONOI REGIONS
        for poly, is_home in regions:
            color = HOME_PLAYER_COLOUR if is_home else AWAY_PLAYER_COLOUR
            x, y = poly.exterior.xy

            mpl_poly = Polygon(
                list(zip(x, y)),
                facecolor=color,
                edgecolor='white',
                alpha=0.4,
                zorder=2
            )

            self.ax.add_patch(mpl_poly)
            self.voronoi_patches.append(mpl_poly)

        # PLAYER COLORS (GK highlighted)
        home_gk, away_gk = self.get_goalkeepers()

        home_colors = [HOME_PLAYER_COLOUR] * len(self.home_coords)
        away_colors = [AWAY_PLAYER_COLOUR] * len(self.away_coords)

        home_sizes = [250] * len(self.home_coords)
        away_sizes = [250] * len(self.away_coords)

        # Highlight goalkeepers
        home_colors[home_gk] = HOME_GK_COLOUR
        away_colors[away_gk] = AWAY_GK_COLOUR

        home_sizes[home_gk] = 380
        away_sizes[away_gk] = 380

        # Remove old scatters safely
        if self.home_scatter:
            self.home_scatter.remove()
        if self.away_scatter:
            self.away_scatter.remove()

        # Draw once only
        self.home_scatter = self.ax.scatter(
            self.home_coords[:,0],
            self.home_coords[:,1],
            c=home_colors,
            s=home_sizes,
            edgecolors='black',
            zorder=10
        )

        self.away_scatter = self.ax.scatter(
            self.away_coords[:,0],
            self.away_coords[:,1],
            c=away_colors,
            s=away_sizes,
            edgecolors='black',
            zorder=10
        )

        for t in self.home_texts + self.away_texts:
            t.remove()

        self.home_texts = []
        self.away_texts = []

        # Player numbers
        for i,(x,y) in enumerate(self.home_coords):
            self.home_texts.append(
                self.ax.text(x,y,str(i+1),
                color='white',ha='center',va='center',
                fontweight='bold',zorder=11)
            )

        for i,(x,y) in enumerate(self.away_coords):
            self.away_texts.append(
                self.ax.text(x,y,str(i+1),
                color='white',ha='center',va='center',
                fontweight='bold',zorder=11)
            )

        # TACTICAL ANALYSIS
        total = self.width * self.height

        home_control = (h_area / total) * 100 if total else 0
        away_control = (a_area / total) * 100 if total else 0

        home_compact = self.analyzer.compactness(self.home_coords)
        away_compact = self.analyzer.compactness(self.away_coords)

        home_width = self.analyzer.width_usage(self.home_coords)
        away_width = self.analyzer.width_usage(self.away_coords)

        center_h, center_a = self.analyzer.central_control(
            self.home_coords, self.away_coords
        )

        over_h, over_a = self.analyzer.overload_score(
            self.home_coords, self.away_coords
        )

        compactness = "Home tighter block" if home_compact < away_compact else "Away tighter block"
        width_analysis = "Home stretching pitch" if home_width > away_width else "Away stretching pitch"
        central_control = "Home dominance" if center_h > center_a else "Away dominance"
        overloads = "Home creating overloads" if over_h > over_a else "Away creating overloads"

        home_form, _, home_row = self.formation_detector.detect_formation_from_player_positions(self.home_coords, team_side="left")
        away_form, _, away_row = self.formation_detector.detect_formation_from_player_positions(self.away_coords, team_side="right")

        home_advice = self.formation_detector.tactical_advice_from_Information_Base(home_row, away_row)
        away_advice = self.formation_detector.tactical_advice_from_Information_Base(away_row, home_row)

        home_mode = home_row['Mode'] if isinstance(home_row, pd.Series) else "Unknown"
        away_mode = away_row['Mode'] if isinstance(away_row, pd.Series) else "Unknown"

        # ---- FORMATIONS ----
        pad_x = 2
        pad_y = 2
        top_y = -pad_y

        home_text = (
            "HOME" +
            f"\nFormation : {home_form}" + 
            (f"\nMode : {home_mode}" if home_mode!="Unknown" else "") +
            "\n\nAWAY" +
            f"\nFormation : {away_form}" +
            (f"\nMode : {away_mode}" if away_mode!="Unknown" else "")
        )

        self.bottom_texts.append(
            self.ax.text(
                pad_x,
                top_y,
                home_text,
                ha="left",
                va="top",
                fontsize=11,
                linespacing=1.6
            )
        )

        # ---- METRICS ----
        control_text = (
            "CONTROL\n"
            f"Home  : {home_control:.1f}%\n"
            f"Away  : {away_control:.1f}%"
        )

        self.bottom_texts.append(
            self.ax.text(
                self.left_w + pad_x,
                top_y,
                control_text,
                ha="left",
                va="top",
                fontsize=11,
                linespacing=1.6
            )
        )

        structure_text = (
            "STRUCTURE\n"
            f"Compactness : {compactness}\n"
            f"Width : {width_analysis}\n"
            f"Central : {central_control}\n"
            f"Overloads : {overloads}"
        )

        self.bottom_texts.append(
            self.ax.text(
                self.left_w + self.mid_w + pad_x,
                top_y,
                structure_text,
                ha="left",
                va="top",
                fontsize=11,
                linespacing=1.6,
                wrap=True
            )
        )

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        click = np.array([event.xdata, event.ydata])
        
        # Check Home Team
        dist_h = np.linalg.norm(self.home_coords - click, axis=1)
        if np.min(dist_h) < 3.0:
            self.selected_point = np.argmin(dist_h)
            self.selected_team = 'home'
            return

        # Check Away Team
        dist_a = np.linalg.norm(self.away_coords - click, axis=1)
        if np.min(dist_a) < 3.0:
            self.selected_point = np.argmin(dist_a)
            self.selected_team = 'away'

    def on_motion(self, event):
        if self.selected_point is None or event.inaxes != self.ax: return
        
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
            self.refresh_graphics() # Full redraw on release
