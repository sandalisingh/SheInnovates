import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from FormationDetector import FormationDetector
from FormationGenerator import FormationGenerator
from InteractiveVoronoiPitch import InteractiveVoronoiPitch
from TacticalAnalyzer import TacticalAnalyzer
from ObjectDetector import ObjectDetector 
import Configurations as CF

class ImageAnalyzer:

    def __init__(self, image_path):
        self.image_path = image_path
        self.device = self._get_device()
        print("Using Device:", self.device)

        self.formation_detector = FormationDetector()
        self.formation_generator = FormationGenerator()
        self.tactical_analyzer = TacticalAnalyzer()

        self.tracking_df = None
        self.annotated_image = None 

        self.team_A_coords = None
        self.team_B_coords = None

        self.team_A_formation = None
        self.team_B_formation = None
        self.team_A_mode = None
        self.team_B_mode = None

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # 1. IMAGE → TRACKING DATAFRAME & VISUALIZATION
    def process_image(self):
        print("\n--- STEP 1: Running Object Detector ---")

        detector = ObjectDetector()
        
        self.tracking_df, self.annotated_image = detector.process_image(
            self.image_path, 
            output_path=CF.OP_IMG_PATH_OBJ_DET 
        )

        if self.tracking_df is None or len(self.tracking_df) == 0:
            raise ValueError("‼️ Detection dataframe is empty. No players found.")

        print("Total players detected:", len(self.tracking_df))
        print(f"✅ Annotated image saved locally as: {CF.OP_IMG_PATH_OBJ_DET}")

        self.show_detected_image()

    def show_detected_image(self):
        """Displays the annotated image with YOLO bounding boxes and team colors"""
        if self.annotated_image is not None:
            img_rgb = cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img_rgb)
            plt.title("Detected Players & Team Classification", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            plt.show(block=False) 
            plt.pause(3) 
            plt.close()

    # 2. EXTRACT TEAM COORDINATES
    def extract_team_coords(self):
        print("\n--- STEP 2: Extracting Team Coordinates ---")

        df = self.tracking_df.copy()
        df.columns = df.columns.str.upper()

        CF.OPPONENT_TEAM_NAME = df["TEAM"].unique()[0]
        CF.MY_TEAM_NAME = df["TEAM"].unique()[1]

        team_A_df = df[df["TEAM"] == CF.MY_TEAM_NAME]
        team_B_df = df[df["TEAM"] == CF.OPPONENT_TEAM_NAME]

        coords_A, ids_A = self.formation_detector.process_centroids(team_A_df)
        coords_B, ids_B = self.formation_detector.process_centroids(team_B_df)

        self.team_A_coords = coords_A
        self.team_B_coords = coords_B

    # 3. FORMATION DETECTION
    def detect_formations(self):
        print("\n--- STEP 3: Detecting Formations ---")

        pattern_A, mode_A, predicted_formation_A = \
            self.formation_detector.detect_formation_from_player_positions(
                self.team_A_coords, team_side="left"
            )

        pattern_B, mode_B, predicted_formation_B = \
            self.formation_detector.detect_formation_from_player_positions(
                self.team_B_coords, team_side="right"
            )

        self.team_A_mode = mode_A
        self.team_B_mode = mode_B

        self.team_A_formation = predicted_formation_A if predicted_formation_A else pattern_A
        self.team_B_formation = predicted_formation_B if predicted_formation_B else pattern_B

        print(f"[{CF.MY_TEAM_NAME}] FORMATION: {self.team_A_formation}")
        print(f"[{CF.MY_TEAM_NAME}] TACTICAL MODE: {self.team_A_mode}")

        print(f"\n[{CF.OPPONENT_TEAM_NAME}] FORMATION: {self.team_B_formation}")
        print(f"[{CF.OPPONENT_TEAM_NAME}] TACTICAL MODE: {self.team_B_mode}")

    # 4. GENERATE TACTICAL TEMPLATES
    def generate_formations(self):
        print("\n--- STEP 4: Generating Tactical Templates ---")

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

    # 5. VULNERABILITY & CONTROL ANALYSIS
    def analyze_and_print_tactics(self, team_A_template, team_B_template):
        print("\n--- STEP 5: Tactical Analysis (Space Control & Vulnerabilities) ---")
        
        # --- 1. Calculate 22-man space for possession control ---
        all_coords = np.vstack((team_A_template, team_B_template))
        epsilon = 1e-6
        all_coords = all_coords + np.random.uniform(-epsilon, epsilon, all_coords.shape)
        
        space_data_22 = self.tactical_analyzer.analyze_space_control(all_coords)
        
        h_area = 0
        a_area = 0
        
        for i, p in enumerate(space_data_22):
            is_home = i < len(team_A_template)
            if is_home: h_area += p['area']
            else: a_area += p['area']
                
        # --- 2. Calculate 11-man space for structural vulnerabilities (FIXED) ---
        home_11_space = self.tactical_analyzer.analyze_space_control(team_A_template)
        away_11_space = self.tactical_analyzer.analyze_space_control(team_B_template)

        home_vuln = self.tactical_analyzer.identify_team_vulnerabilities(home_11_space, 'home', CF.MY_TEAM_NAME)
        away_vuln = self.tactical_analyzer.identify_team_vulnerabilities(away_11_space, 'away', CF.OPPONENT_TEAM_NAME)

        vulnerabilities = home_vuln + away_vuln
        vulnerabilities.sort(key=lambda v: v['area'], reverse=True)
        
        # Print Control Percentages
        total_pitch_area = CF.PITCH_LENGTH * CF.PITCH_WIDTH
        print(f"\n>> SPACE CONTROL:")
        print(f"   {CF.MY_TEAM_NAME}: {(h_area / total_pitch_area) * 100:.1f}%")
        print(f"   {CF.OPPONENT_TEAM_NAME}: {(a_area / total_pitch_area) * 100:.1f}%")

        # Print Vulnerabilities
        print(f"\n>> STRUCTURAL VULNERABILITIES:")
        if not vulnerabilities:
            print("   ✅ No major spatial gaps detected in the central zones.")
        else:
            for v in vulnerabilities[:4]: # Print top 4 gaps
                print(f"   🚨 {v['detail']}")

    # FULL PIPELINE
    def run(self):
        self.process_image()
        self.extract_team_coords()
        self.detect_formations()
        
        team_A_template, team_B_template = self.generate_formations()
        self.analyze_and_print_tactics(team_A_template, team_B_template)

        print("\n--- STEP 6: Launching Interactive Tactical Board ---")
        
        InteractiveVoronoiPitch(
            team_A_template, team_B_template, 
            self.team_A_formation, self.team_B_formation,
            self.team_A_mode, self.team_B_mode
        )
        plt.show()