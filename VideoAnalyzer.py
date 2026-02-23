import torch
import matplotlib.pyplot as plt

from FormationDetector import FormationDetector
from FormationGenerator import FormationGenerator
from InteractiveVoronoiPitch import InteractiveVoronoiPitch
from ObjectDetector import ObjectDetector
import Configurations as CF


class VideoAnalyzer:

    def __init__(self, video_path=CF.IP_VID_PATH_OBJ_DET):
        self.video_path = video_path

        self.device = self._get_device()
        print("Using Device:", self.device)

        self.formation_detector = FormationDetector()
        self.formation_generator = FormationGenerator()

        # Will store tracking dataframe returned by ObjectDetection
        self.tracking_df = None

        # Per-team centroid positions
        self.team_A_coords = None
        self.team_B_coords = None

        # Detected formations
        self.team_A_formation = None
        self.team_B_formation = None
        self.team_A_mode = None
        self.team_B_mode = None

    # DEVICE SELECTION
    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # VIDEO → TRACKING DATAFRAME
    def process_video(self):
        print("\nRunning Object Detector and Tracking...")

        detector = ObjectDetector()
        self.tracking_df = detector.process_video(self.video_path) # Or detector.process_image() for the ImageAnalyzer

        if self.tracking_df is None or len(self.tracking_df) == 0:
            raise ValueError("‼️ Tracking dataframe is empty.")

        print("Total tracking rows:", len(self.tracking_df))

    # COMPUTE PLAYER CENTROIDS PER TEAM
    def compute_team_centroids(self):

        """
        For each team:
        1. Filter dataframe
        2. Use FormationDetector.process_centroids()
        3. Get mean X,Y for each player across all frames

        This produces stable average player positions
        for formation detection.
        """

        print("\nComputing mean centroids per team...")

        # Ensure consistent column format
        df = self.tracking_df.copy()
        df.columns = df.columns.str.upper()

        CF.OPPONENT_TEAM_NAME = df["TEAM"].unique()[0]
        CF.MY_TEAM_NAME = df["TEAM"].unique()[1]

        print("MY TEAM NAME = ", CF.MY_TEAM_NAME)
        print("OPPONENT NAME = ", CF.OPPONENT_TEAM_NAME)

        # Separate teams
        team_A_df = df[df["TEAM"] == CF.MY_TEAM_NAME]
        team_B_df = df[df["TEAM"] == CF.OPPONENT_TEAM_NAME]

        # Process centroids using FormationDetector helper
        coords_A, ids_A = self.formation_detector.process_centroids(team_A_df)
        coords_B, ids_B = self.formation_detector.process_centroids(team_B_df)

        self.team_A_coords = coords_A
        self.team_B_coords = coords_B

        print("Team A players detected:", len(coords_A))
        print("Team B players detected:", len(coords_B))

    # FORMATION DETECTION FOR BOTH TEAMS
    def detect_formations(self):

        """
        Uses FormationDetector.detect_formation_from_player_positions()

        Inputs:
        - Team centroid coordinates
        - Team side (left/right)

        Automatically:
        - Removes GK (deepest player)
        - Detects tactical lines
        - Maps to ML Roles (Hungarian Alg)
        - Predicts Base Structure using KNN
        """

        print("\nDetecting formations...")

        # Team A assumed left side
        # Unpacks the predicted string directly from the ML model
        pattern_A, mode_A, predicted_formation_A = \
            self.formation_detector.detect_formation_from_player_positions(
                self.team_A_coords,
                team_side="left"
            )

        # Team B assumed right side
        pattern_B, mode_B, predicted_formation_B = \
            self.formation_detector.detect_formation_from_player_positions(
                self.team_B_coords,
                team_side="right"
            )

        self.team_A_mode = mode_A
        self.team_B_mode = mode_B

        # Fallback to pattern string if ML somehow fails
        self.team_A_formation = predicted_formation_A if predicted_formation_A else pattern_A
        self.team_B_formation = predicted_formation_B if predicted_formation_B else pattern_B

        print("\nTEAM A FORMATION:", self.team_A_formation)
        print("TEAM A MODE:", self.team_A_mode)

        print("\nTEAM B FORMATION:", self.team_B_formation)
        print("TEAM B MODE:", self.team_B_mode)

    # GENERATE TACTICAL TEMPLATES
    def generate_formations(self):

        """
        Uses FormationGenerator to create tactical template coordinates
        for both teams using detected formation + mode.
        """

        print("\nGenerating tactical templates...")

        # Pass the full ML predicted string. 
        # The updated FormationGenerator will intelligently parse out Structure, Mode, and Shape.
        team_A_template = self.formation_generator.generate_template_from_formation(
            formation_name=self.team_A_formation,
            mode=self.team_A_mode,
            team_side="left"
        )

        team_B_template = self.formation_generator.generate_template_from_formation(
            formation_name=self.team_B_formation,
            mode=self.team_B_mode,
            team_side="right"
        )

        return team_A_template, team_B_template

    # FULL PIPELINE
    def run(self):

        """
        Full pipeline:

        1. Run object detection + tracking
        2. Compute per-player centroids across video
        3. Detect formations for both teams
        4. Generate tactical templates
        5. Launch Voronoi tactical visualization
        """

        self.process_video()
        self.compute_team_centroids()
        self.detect_formations()

        team_A_template, team_B_template = self.generate_formations()

        app = InteractiveVoronoiPitch(
            team_A_template, team_B_template,
            self.team_A_formation, self.team_B_formation,
            self.team_A_mode, self.team_B_mode
        )
        app.run()
