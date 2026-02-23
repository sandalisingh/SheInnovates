from enum import Enum

# -------------------------- PATHS --------------------------
YOLO_MODEL_PATH = "Models/best_object_detection_model.pt"
FORMATION_CLASSIFIER_PATH = "Models/knn_formation_model.pkl"
FORMATIONS_INFO_DB = "Data/Formations_info_transformed.csv"
COUNTER_FORMATION_PREDICTOR_PATH = "Models/counter_form_predictor.pkl"
COUNTER_FORMATION_TARGET_LABEL_ENCODER_PATH = "Models/counter_form_target_label_encoder.pkl"

# -------------------------- OBJECT DETECTION --------------------------

MY_TEAM_NAME = "Home"
OPPONENT_TEAM_NAME = "Away"

IP_IMG_PATH_OBJ_DET = "Data/Raw/input_obj_det.png"
OP_IMG_PATH_OBJ_DET = "Data/Raw/output_obj_det.png"
IP_VID_PATH_OBJ_DET = "Data/Raw/input_video_obj_det.mp4"
OP_VID_PATH_OBJ_DET = "Data/Raw/output_video_obj_det.mp4"

# Class Mapping (Verify with your model.names)
CLASS_MAP = {
    'ball': 0,
    'player': 1,
    'referee': 2,
    'goalkeeper': 3 # If your model detects GK separately
}

# Colors (BGR)
COLORS = {
    'ball': (0, 255, 255),       # Yellow
    'referee': (0, 0, 0),        # Black
    'goalkeeper': (255, 0, 255), # Magenta
    'team_0': (255, 0, 0),       # Blue
    'team_1': (0, 0, 255)        # Red
}

# -------------------------- DRAWING PITCH --------------------------

PITCH_LENGTH = 105
PITCH_WIDTH = 68
PITCH_COLOUR = "#358751"

AREA_CAPTURE_MIN_THRESHOLD = 550 # in m^2

HOME_PLAYER_COLOUR = '#3498db'
HOME_GK_COLOUR = "#0ff1b5"

AWAY_PLAYER_COLOUR = '#e74c3c'
AWAY_GK_COLOUR = '#e67e22'

# -------------------------- FORMATION STRUCTURES --------------------------

class Formation(Enum):

    # 3 at the back
    F_3_1_4_2 = "3-1-4-2"
    F_3_4_1_2 = "3-4-1-2"
    F_3_4_2_1 = "3-4-2-1"
    F_3_4_3   = "3-4-3"
    F_3_5_1_1 = "3-5-1-1"
    F_3_5_2   = "3-5-2"

    # 4 at the back
    F_4_1_2_1_2 = "4-1-2-1-2"
    F_4_1_3_2   = "4-1-3-2"
    F_4_1_4_1   = "4-1-4-1"
    F_4_2_2_2   = "4-2-2-2"
    F_4_2_3_1   = "4-2-3-1"
    F_4_2_4     = "4-2-4"
    F_4_3_1_2   = "4-3-1-2"
    F_4_3_2_1   = "4-3-2-1"
    F_4_3_3     = "4-3-3"
    F_4_4_1_1   = "4-4-1-1"
    F_4_4_2     = "4-4-2"
    F_4_5_1     = "4-5-1"

    # 5 at the back
    F_5_2_1_2 = "5-2-1-2"
    F_5_2_2_1 = "5-2-2-1"
    F_5_2_3   = "5-2-3"
    F_5_3_2   = "5-3-2"
    F_5_4_1   = "5-4-1"

# -------------------------- FORMATION MODES --------------------------

class Mode(Enum):
    ATTACKING = "Attacking"
    DEFENDING = "Defending"
    MIDFIELD  = "Midfield"
    HOLDING   = "Holding"
    DEFENSIVE = "Defensive"
    BALANCED  = "Balanced"
    UNKNOWN   = "Unknown"

# -------------------------- FORMATION SHAPES --------------------------

class Shape(Enum):
    NA        = ""
    DIAMOND   = "Diamond"
    FLAT      = "Flat"
    NARROW    = "Narrow"
    WIDE      = "Wide"
    ATTACK    = "Attack"
    DEFEND    = "Defend"
    HOLDING   = "Holding"
    FALSE_9   = "False 9"
    VARIANT_2 = "(2)"
    VARIANT_3 = "(3)"
    VARIANT_4 = "(4)"
    VARIANT_5 = "(5)"

# -------------------------- PLAYER ROLES --------------------------

class Role(Enum):

    # Goalkeeper
    Goalkeeper = "GK"

    # Defenders
    Left_Center_Back = "LCB"
    Center_Back = "CB"
    Right_Center_Back = "RCB"
    Left_Back = "LB"
    Right_Back = "RB"
    Left_Wing_Back = "LWB"
    Right_Wing_Back = "RWB"

    # Defensive Midfield
    Central_Defensive_Midfielder = "CDM"
    Left_Defensive_Midfielder = "LDM"
    Right_Defensive_Midfielder = "RDM"

    # Central Midfield
    Central_Midfielder = "CM"
    Left_Central_Midfielder = "LCM"
    Right_Central_Midfielder = "RCM"

    # Attacking Midfield
    Central_Attacking_Midfielder = "CAM"
    Left_Attacking_Midfielder = "LAM"
    Right_Attacking_Midfielder = "RAM"

    # Wide Midfield / Wingers
    Left_Midfielder = "LM"
    Right_Midfielder = "RM"
    Left_Winger = "LW"
    Right_Winger = "RW"

    # Forwards
    Center_Forward = "CF"
    Left_Forward = "LF"
    Right_Forward = "RF"

    # Strikers
    Striker = "ST"
    Left_Striker = "LS"
    Right_Striker = "RS"

ROLE_LIST = list(Role)
ROLE_VALUE_LIST = list(r.value for r in Role)

# ========================================== AMPUTEE FOOTBALL SPECIFICS ==========================================

# Separate Paths
FORMATION_CLASSIFIER_PATH_AMPUTEE = "Models/knn_formation_model_amputee.pkl"
FORMATIONS_INFO_DB_AMPUTEE = "Data/Amputee_Formations_info.csv"

# Amputee 7-a-side Formations
class AmputeeFormation(Enum):
    F_2_3_1   = "2-3-1"
    F_3_2_1   = "3-2-1"
    F_2_2_2   = "2-2-2"
    F_1_3_2   = "1-3-2"
    F_2_1_2_1 = "2-1-2-1"
    F_1_4_1   = "1-4-1"