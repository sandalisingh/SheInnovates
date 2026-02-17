FORMATIONS_INFO_DB = "Data/Formations_info_transformed.csv"

PITCH_LENGTH = 105
PITCH_WIDTH = 68
PITCH_COLOUR = "#358751"

AREA_CAPTURE_MIN_THRESHOLD = 550 # in m^2

MY_TEAM_NAME = "Home"
OPPONENT_TEAM_NAME = "Away"

HOME_PLAYER_COLOUR = '#3498db'
HOME_GK_COLOUR = "#0ff1b5"

AWAY_PLAYER_COLOUR = '#e74c3c'
AWAY_GK_COLOUR = '#e67e22'

MODE_ATTACKING = "Attacking"
MODE_DEFENSIVE = "Defensive"
MODE_BALANCED = "Balanced"
MODE_UNKNOWN = "Unknown"

ROLE_Goalkeeper = "GK"

ROLE_Center_Back = "CB"
ROLE_Left_Center_Back = "LCB"
ROLE_Right_Center_Back = "RCB"
ROLE_Left_Back = "LB"
ROLE_Right_Back = "RB"

ROLE_Central_Midfielder = "CM"
ROLE_Left_Central_Midfielder = "LCM"
ROLE_Right_Central_Midfielder = "RCM"

ROLE_Left_Midfielder = "LM"
ROLE_Right_Midfielder = "RM"

ROLE_Central_Attacking_Midfielder = "CAM"
ROLE_Left_Attacking_Midfielder = "LAM"
ROLE_Right_Attacking_Midfielder = "RAM"

ROLE_Central_Defensive_Midfielder = "CDM"
ROLE_Left_Defensive_Midfielder = "LDM"
ROLE_Right_Defensive_Midfielder = "RDM"

ROLE_Left_Winger= "LW"
ROLE_Right_Winger = "RW"

ROLE_Left_Wing_Back= "LWB"
ROLE_Right_Wing_Back = "RWB"

ROLE_Center_Forward = "CF"
ROLE_Left_Forward = "LF"
ROLE_Right_Forward = "RF"

ROLE_Striker = "ST"
ROLE_Left_Striker = "LS"
ROLE_Right_Striker = "RS"

ROLE_LIST = [

    # Goalkeeper
    ROLE_Goalkeeper,

    # Defenders
    ROLE_Left_Center_Back,
    ROLE_Center_Back,
    ROLE_Right_Center_Back,
    ROLE_Left_Back,
    ROLE_Right_Back,
    ROLE_Left_Wing_Back,
    ROLE_Right_Wing_Back,

    # Defensive Midfield
    ROLE_Central_Defensive_Midfielder,
    ROLE_Left_Defensive_Midfielder,
    ROLE_Right_Defensive_Midfielder,

    # Central Midfield
    ROLE_Central_Midfielder,
    ROLE_Left_Central_Midfielder,
    ROLE_Right_Central_Midfielder,

    # Attacking Midfield
    ROLE_Central_Attacking_Midfielder,
    ROLE_Left_Attacking_Midfielder,
    ROLE_Right_Attacking_Midfielder,

    # Wide Midfield / Wingers
    ROLE_Left_Midfielder,
    ROLE_Right_Midfielder,
    ROLE_Left_Winger,
    ROLE_Right_Winger,

    # Forwards
    ROLE_Center_Forward,
    ROLE_Left_Forward,
    ROLE_Right_Forward,

    # Strikers
    ROLE_Striker,
    ROLE_Left_Striker,
    ROLE_Right_Striker
]
