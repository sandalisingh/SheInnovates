from InteractiveVoronoiPitch import InteractiveVoronoiPitch
import matplotlib.pyplot as plt
from FormationGenerator import FormationGenerator
import Configurations as CF

MY_TEAM_NAME = "A"
my_formation = "4-3-3"
my_mode = CF.MODE_ATTACKING

OPPONENT_TEAM_NAME = "B"
opp_formation = "4-4-2"
opp_mode = CF.MODE_BALANCED

formationGenerator = FormationGenerator()

team_A = formationGenerator.generate_template_from_formation(my_formation, "left", my_mode)
team_B = formationGenerator.generate_template_from_formation(opp_formation, "right", opp_mode)

app = InteractiveVoronoiPitch(team_A, team_B)
plt.show()