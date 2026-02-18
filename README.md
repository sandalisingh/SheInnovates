# ⚽ AI Football Tactical Analyzer

An end-to-end computer vision and machine learning pipeline that transforms raw football (soccer) video broadcasts into an **interactive, real-time tactical board**.

This project automatically detects players, clusters them into tactical lines, maps their spatial coordinates to traditional football roles using the Hungarian algorithm, and predicts the team's overarching formation and tactical phase. Finally, it generates an interactive Voronoi diagram to analyze space control, team compactness, and structural vulnerabilities.



## Key Features

* **Automated Video Tracking:** Uses object detection to identify players, sample jersey colors to separate teams, and track movements across frames.
* **ML Formation Detection:** Uses **K-Means Clustering** to identify defensive, midfield, and attacking lines.
* Applies the **Hungarian Algorithm** to perfectly map continuous  player coordinates to 27 discrete, ideal football roles (e.g., LCB, CAM, False 9).
* Uses a trained **K-Nearest Neighbors (KNN)** model to predict the exact Formation Structure and Shape (e.g., `"3-4-3 Diamond"`).


* **Tactical Phase Recognition:** Analyzes the depth of the team block relative to the halfway line to automatically classify the team's current phase as *Attacking*, *Defending*, or *Balanced*.
* **Interactive Voronoi Pitch:** A fully interactive 2D GUI built with Matplotlib.

* **Vulnerability Detection:** Highlights players in the central zones who are covering dangerously large amounts of space (Sparse Coverage).
* **Structural Stats:** Calculates Team Compactness, Width Usage, Central Control, and Local Overloads.
* **Counter-Tactics:** Suggests optimal counter-formations based on a custom tactical database.



## System Architecture

1. **`ObjectDetector.py`**: Ingests video, detects players using YOLO model, assigns teams via color sampling and k-means, and outputs a DataFrame of  bounding box coordinates.
2. **`VideoAnalyzer.py`**: The orchestrator. Averages player coordinates across frames to find stable tactical centroids.
3. **`FormationDetector.py`**: Normalizes coordinates to a  meter pitch. Uses grid-based spatial matching and KNN to classify the formation of the teams.
4. **`FormationGenerator.py`**: Acts as the tactical blueprint engine. Takes the predicted formation (e.g., "4-3-3 Attacking") and generates the ideal formation template.
4. **`TacticalAnalyzer.py`**: Analyzes the vulnerabilities and structural stats (like central control) for both the teams.
5. **`InteractiveVoronoiPitch.py`**: Renders the dynamic UI, calculates SciPy Voronoi regions, Shapely polygon intersections, and generates the strategic advice panel.

## Getting Started

### Prerequisites

* Python 3.8+
* Recommended to use a virtual environment (`venv` or `conda`).

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sandalisingh/SheInnovates.git
cd SheInnovates

```


2. **Install dependencies:**
Ensure you have PyTorch installed for your specific hardware (CUDA/MPS/CPU), then install the rest:
```bash
pip install pandas numpy matplotlib scipy shapely scikit-learn opencv-python torch

```


3. **Prepare the Data & Models:**
* Ensure your video file is placed in the correct path specified in `Configurations.py` (e.g., `CF.IP_VID_PATH_OBJ_DET`).
* Ensure the following are present - 
    * `Data/Formations_info_transformed.csv` 
    * `Models/knn_formation_model.pkl`
    * `Models/best_object_detection_model.pt`



### Running the Analyzer

To kick off the full pipeline (Video Processing  ML Detection  Interactive GUI):

```bash
python3 Video_analysis_testing.py

```

## Built With

* **[PyTorch](https://pytorch.org/) & OpenCV** - Computer Vision & Object Tracking
* **[Scikit-Learn](https://scikit-learn.org/)** - K-Means, KNN
* **[SciPy](https://scipy.org/)** - Linear Sum Assignment (Hungarian Algorithm) & Voronoi Tessellation
* **[Shapely](https://shapely.readthedocs.io/)** - Polygon intersection and area calculations
* **[Matplotlib](https://matplotlib.org/)** - Interactive 2D graphical rendering

## Future Roadmap

* **Homography Integration:** Implement OpenCV perspective transformations to map broadcast camera angles directly to a flat 2D top-down view before centroid calculation.
* **Temporal Smoothing:** Add Kalman filters to track player movement trajectories over time rather than just static frame averaging.
* **Player ID Tracking:** Integrate OCR to read jersey numbers for specific player analysis.