import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import box
from dash import Dash, dcc, html, Input, Output, State

from FormationDetector import FormationDetector, get_info_for_formation
from TacticalAnalyzer import TacticalAnalyzer
import Configurations as CF
from Counter_formation_predictor import predict_counter_strategy

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

    def __init__(self, home_coords, away_coords,
                 home_form_str, away_form_str,
                 home_mode, away_mode):

        self.width = 105
        self.height = 68

        self.formation_detector = FormationDetector()
        self.tactical_analyzer = TacticalAnalyzer()

        self.home_coords = np.array(home_coords, dtype=float)
        self.away_coords = np.array(away_coords, dtype=float)

        self.home_form_str = home_form_str
        self.away_form_str = away_form_str
        self.home_mode = home_mode
        self.away_mode = away_mode

        self.pitch_poly = box(0, 0, self.width, self.height)

        self.app = Dash(__name__)
        self.build_layout()
        self.register_callbacks()

    # =============================
    # PITCH LINES
    # =============================
    def draw_pitch_lines(self, fig):
        # Using a semi-transparent black for pitch lines to heavily contrast with white Voronoi lines
        line_dict = dict(color="rgba(0, 0, 0, 0.4)", width=2.5) 
        
        # Outer boundary
        fig.add_trace(go.Scatter(x=[0, self.width, self.width, 0, 0], y=[0, 0, self.height, self.height, 0], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        # Halfway line
        fig.add_trace(go.Scatter(x=[self.width/2, self.width/2], y=[0, self.height], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        
        # Center circle
        theta = np.linspace(0, 2*np.pi, 100)
        cx, cy, r = self.width/2, self.height/2, 9.15
        fig.add_trace(go.Scatter(x=cx + r*np.cos(theta), y=cy + r*np.sin(theta), mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        
        # Penalty Boxes
        fig.add_trace(go.Scatter(x=[0, 16.5, 16.5, 0], y=[self.height/2 - 20.16, self.height/2 - 20.16, self.height/2 + 20.16, self.height/2 + 20.16], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[self.width, self.width-16.5, self.width-16.5, self.width], y=[self.height/2 - 20.16, self.height/2 - 20.16, self.height/2 + 20.16, self.height/2 + 20.16], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        
        # 6 yard boxes
        fig.add_trace(go.Scatter(x=[0, 5.5, 5.5, 0], y=[self.height/2 - 9.16, self.height/2 - 9.16, self.height/2 + 9.16, self.height/2 + 9.16], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[self.width, self.width-5.5, self.width-5.5, self.width], y=[self.height/2 - 9.16, self.height/2 - 9.16, self.height/2 + 9.16, self.height/2 + 9.16], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        
        # Goals
        fig.add_trace(go.Scatter(x=[-2, 0, 0, -2, -2], y=[self.height/2 - 3.66, self.height/2 - 3.66, self.height/2 + 3.66, self.height/2 + 3.66, self.height/2 - 3.66], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[self.width+2, self.width, self.width, self.width+2, self.width+2], y=[self.height/2 - 3.66, self.height/2 - 3.66, self.height/2 + 3.66, self.height/2 + 3.66, self.height/2 - 3.66], mode="lines", line=line_dict, hoverinfo="skip", showlegend=False))
        
        # Penalty spots
        fig.add_trace(go.Scatter(x=[11, self.width-11], y=[self.height/2, self.height/2], mode="markers", marker=dict(color="rgba(0,0,0,0.4)", size=6), hoverinfo="skip", showlegend=False))

    # =============================
    # BUILD FIGURE & ANALYSIS PANEL
    # =============================
    def build_figure_and_analysis(self, home_coords, away_coords, home_form, away_form, home_mode, away_mode):

        fig = go.Figure()

        # Margins significantly reduced because text is no longer overlaying the plot
        fig.update_layout(
            plot_bgcolor=CF.PITCH_COLOUR,
            paper_bgcolor="#111111", # Dark background outside the pitch
            dragmode=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, self.width], visible=False, fixedrange=True, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[0, self.height], visible=False, fixedrange=True),
        )

        self.draw_pitch_lines(fig)

        # === COMPUTE VORONOI & AREA ===
        all_points = np.vstack((home_coords, away_coords))
        epsilon = 1e-6 
        all_points = all_points + np.random.uniform(-epsilon, epsilon, all_points.shape)
        vor = Voronoi(all_points)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        home_space = self.tactical_analyzer.analyze_space_control(home_coords)
        away_space = self.tactical_analyzer.analyze_space_control(away_coords)
        home_vuln = self.tactical_analyzer.identify_team_vulnerabilities(home_space, 'home', CF.MY_TEAM_NAME)
        away_vuln = self.tactical_analyzer.identify_team_vulnerabilities(away_space, 'away', CF.OPPONENT_TEAM_NAME)
        vulnerabilities = home_vuln + away_vuln
        vulnerabilities.sort(key=lambda v: v['area'], reverse=True)

        h_area = 0
        a_area = 0

        # Draw Voronoi 
        for i, region in enumerate(regions):
            polygon = vertices[region]
            poly = ShapelyPoly(polygon)
            inter = poly.intersection(self.pitch_poly)

            if inter.is_empty:
                continue

            is_home = i < len(home_coords)
            team_str = 'home' if is_home else 'away'
            player_idx = (i + 1) if is_home else (i - len(home_coords) + 1)

            if is_home: h_area += inter.area
            else: a_area += inter.area

            is_vulnerable = any(v['player_id'] == player_idx and v['team_tag'] == team_str for v in vulnerabilities)
            alpha = 0.85 if is_vulnerable else 0.35
            color = CF.HOME_PLAYER_COLOUR if is_home else CF.AWAY_PLAYER_COLOUR

            geoms = [inter] if inter.geom_type == 'Polygon' else inter.geoms
            for p in geoms:
                x, y = map(list, p.exterior.xy)
                fig.add_trace(go.Scatter(
                    x=x, y=y, fill="toself", mode="lines",
                    line=dict(color="white", width=1.5),
                    fillcolor=color, opacity=alpha,
                    hoverinfo="skip", showlegend=False
                ))

        # === METRICS FOR TEXTUAL ANALYSIS ===
        total_area = self.width * self.height
        home_control = (h_area / total_area) * 100 if total_area else 0
        away_control = (a_area / total_area) * 100 if total_area else 0

        home_compact = self.tactical_analyzer.compactness(home_coords)
        away_compact = self.tactical_analyzer.compactness(away_coords)
        home_width = self.tactical_analyzer.width_usage(home_coords)
        away_width = self.tactical_analyzer.width_usage(away_coords)
        center_h, center_a = self.tactical_analyzer.central_control(home_coords, away_coords)
        over_h, over_a = self.tactical_analyzer.overload_score(home_coords, away_coords)

        compactness = f"{CF.MY_TEAM_NAME} tighter block" if home_compact < away_compact else f"{CF.OPPONENT_TEAM_NAME} tighter block"
        width_analysis = f"{CF.MY_TEAM_NAME} stretching pitch" if home_width > away_width else f"{CF.OPPONENT_TEAM_NAME} stretching pitch"
        central_control = f"{CF.MY_TEAM_NAME} dominance" if center_h > center_a else f"{CF.OPPONENT_TEAM_NAME} dominance"
        overloads = f"{CF.MY_TEAM_NAME} creating overloads" if over_h > over_a else f"{CF.OPPONENT_TEAM_NAME} creating overloads"

        counter_formations = "N/A"
        try:
            away_form_info = get_info_for_formation(away_form)
            if away_form_info and 'Structure' in away_form_info:
                counters, _ = predict_counter_strategy(away_form_info['Structure'], away_form_info['Shape'], away_form_info['Mode'])
                if counters: counter_formations = counters
        except Exception:
            pass
        
        home_display = "Unclear" if not home_form or "Unclear" in home_form else str(home_form).replace(str(home_mode), "").strip()
        away_display = "Unclear" if not away_form or "Unclear" in away_form else str(away_form).replace(str(away_mode), "").strip()

        # === CONSTRUCT HTML SIDE PANEL ===
        analysis_div = html.Div([
            # Home Team Info
            html.Span(f"{CF.MY_TEAM_NAME.upper()}", style={"color": CF.HOME_PLAYER_COLOUR, "fontSize": "16px", "fontWeight": "bold"}), html.Br(),
            html.Span(f"Form: {home_display}"), html.Br(),
            html.Span(f"Mode: {home_mode}"), html.Br(), html.Br(),
            
            # Away Team Info
            html.Span(f"{CF.OPPONENT_TEAM_NAME.upper()}", style={"color": CF.AWAY_PLAYER_COLOUR, "fontSize": "16px", "fontWeight": "bold"}), html.Br(),
            html.Span(f"Form: {away_display}"), html.Br(),
            html.Span(f"Mode: {away_mode}"), html.Br(),
            
            html.Hr(style={"borderTop": "1px solid #444", "margin": "15px 0"}),
            
            # Space Control
            html.Span("SPACE CONTROL", style={"color": "#FFF", "fontSize": "14px", "fontWeight": "bold"}), html.Br(),
            html.Span(f"{CF.MY_TEAM_NAME}: {home_control:.1f}%"), html.Br(),
            html.Span(f"{CF.OPPONENT_TEAM_NAME}: {away_control:.1f}%"), html.Br(), html.Br(),
            
            # Counters
            html.Span(f"COUNTERS TO {CF.OPPONENT_TEAM_NAME}", style={"color": "#FFF", "fontSize": "14px", "fontWeight": "bold"}), html.Br(),
            html.Span(f"{counter_formations}"), html.Br(),
            
            html.Hr(style={"borderTop": "1px solid #444", "margin": "15px 0"}),
            
            # Structure
            html.Span("STRUCTURE", style={"color": "#FFF", "fontSize": "14px", "fontWeight": "bold"}), html.Br(),
            html.Span(f"Compactness: {compactness}"), html.Br(),
            html.Span(f"Width: {width_analysis}"), html.Br(),
            html.Span(f"Central: {central_control}"), html.Br(),
            html.Span(f"Overloads: {overloads}"), html.Br(), html.Br(),
            
        ], style={"fontFamily": "Arial, sans-serif", "color": "#E0E0E0", "fontSize": "13px", "lineHeight": "1.5"})

        if vulnerabilities:
            vuln_div = [
                html.Span("WEAKNESSES", style={"color": "#ff4d4d", "fontSize": "14px", "fontWeight": "bold"}), html.Br()
            ]
            for v in vulnerabilities[:5]: # Show top 5 weaknesses
                vuln_div.extend([html.Span(f"• {v['detail']}"), html.Br()])
            
            analysis_div.children.extend(vuln_div)


        # === DRAGGABLE PLAYERS (Rendered as Shapes) ===
        shapes = []
        home_gk = np.argmin(home_coords[:, 0])
        away_gk = np.argmax(away_coords[:, 0])

        for i, (x, y) in enumerate(home_coords):
            r = 1.6 if i == home_gk else 1.2
            color = CF.HOME_GK_COLOUR if i == home_gk else CF.HOME_PLAYER_COLOUR
            shapes.append(dict(
                type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                fillcolor=color, line=dict(color="black", width=1.5),
                label=dict(text=str(i+1), font=dict(color="white", size=11, family="Arial Black"))
            ))

        for i, (x, y) in enumerate(away_coords):
            r = 1.6 if i == away_gk else 1.2
            color = CF.AWAY_GK_COLOUR if i == away_gk else CF.AWAY_PLAYER_COLOUR
            shapes.append(dict(
                type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                fillcolor=color, line=dict(color="black", width=1.5),
                label=dict(text=str(i+1), font=dict(color="white", size=11, family="Arial Black"))
            ))

        fig.update_layout(shapes=shapes)
        return fig, analysis_div

    # =============================
    # LAYOUT
    # =============================
    def build_layout(self):
        # Generate initial state 
        fig, analysis_html = self.build_figure_and_analysis(
            self.home_coords, self.away_coords, 
            self.home_form_str, self.away_form_str, 
            self.home_mode, self.away_mode
        )
        
        self.app.layout = html.Div([
            dcc.Store(id="player-store",
                data={
                    "home": self.home_coords.tolist(),
                    "away": self.away_coords.tolist(),
                    "home_form": self.home_form_str,
                    "away_form": self.away_form_str,
                    "home_mode": self.home_mode,
                    "away_mode": self.away_mode
                }),
                
            html.Div([
                # Left side: The Interactive Graph
                dcc.Graph(
                    id="pitch",
                    figure=fig,
                    style={"flex": "1", "height": "95vh"}, # Fills the left area
                    config={
                        "editable": True, 
                        "edits": {
                            "shapePosition": True,  
                            "annotationPosition": False,
                            "legendPosition": False,
                            "titleText": False,
                        },
                        "displayModeBar": False
                    }
                ),
                # Right side: The Dedicated Sidebar for Text Panel
                html.Div(
                    id="analysis-panel",
                    children=analysis_html,
                    style={
                        "width": "350px", 
                        "minWidth": "350px",
                        "backgroundColor": "#1A1A1A", 
                        "padding": "25px", 
                        "height": "95vh",
                        "overflowY": "auto",
                        "boxSizing": "border-box",
                        "borderLeft": "3px solid #333"
                    }
                )
            ], style={"display": "flex", "flexDirection": "row", "width": "100%", "backgroundColor": "#111111"})
        ])

    # =============================
    # CALLBACK
    # =============================
    def register_callbacks(self):

        @self.app.callback(
            Output("pitch", "figure"),
            Output("analysis-panel", "children"),
            Output("player-store", "data"),
            Input("pitch", "relayoutData"),
            State("player-store", "data")
        )
        def update_on_drag(relayoutData, store):

            home = np.array(store["home"])
            away = np.array(store["away"])
            h_form = store.get("home_form", self.home_form_str)
            a_form = store.get("away_form", self.away_form_str)
            h_mode = store.get("home_mode", self.home_mode)
            a_mode = store.get("away_mode", self.away_mode)

            # Check if drag event happened
            if not relayoutData:
                fig, analysis_div = self.build_figure_and_analysis(home, away, h_form, a_form, h_mode, a_mode)
                return fig, analysis_div, store

            updated = False

            if 'shapes' in relayoutData:
                for idx, shape in enumerate(relayoutData['shapes']):
                    cx = (shape.get('x0', 0) + shape.get('x1', 0)) / 2
                    cy = (shape.get('y0', 0) + shape.get('y1', 0)) / 2
                    
                    if idx < 11: home[idx] = [cx, cy]
                    else: away[idx - 11] = [cx, cy]
                updated = True
            else:
                updates = {}
                for key, val in relayoutData.items():
                    if key.startswith('shapes['):
                        idx = int(key.split('[')[1].split(']')[0])
                        prop = key.split('.')[1] 
                        if idx not in updates: updates[idx] = {}
                        updates[idx][prop] = val

                for idx, props in updates.items():
                    if idx < 11: cx, cy = home[idx]
                    else: cx, cy = away[idx - 11]
                        
                    if 'x0' in props and 'x1' in props: cx = (props['x0'] + props['x1']) / 2
                    if 'y0' in props and 'y1' in props: cy = (props['y0'] + props['y1']) / 2
                        
                    cx = np.clip(cx, 0, self.width)
                    cy = np.clip(cy, 0, self.height)

                    if idx < 11: home[idx] = [cx, cy]
                    else: away[idx - 11] = [cx, cy]
                    updated = True

            if updated:
                try:
                    _, h_mode, h_form = self.formation_detector.detect_formation_from_player_positions(home, team_side="left")
                except Exception:
                    h_mode, h_form = "Unclear", "Unclear"
                    
                try:
                    _, a_mode, a_form = self.formation_detector.detect_formation_from_player_positions(away, team_side="right")
                except Exception:
                    a_mode, a_form = "Unclear", "Unclear"
                
                store["home"] = home.tolist()
                store["away"] = away.tolist()
                store["home_form"] = str(h_form)
                store["away_form"] = str(a_form)
                store["home_mode"] = str(h_mode)
                store["away_mode"] = str(a_mode)

            fig, analysis_div = self.build_figure_and_analysis(home, away, h_form, a_form, h_mode, a_mode)
            return fig, analysis_div, store
          
    # =============================
    # RUN
    # =============================
    def run(self):
        self.app.run(debug=True)