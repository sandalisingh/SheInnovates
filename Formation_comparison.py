from InteractiveVoronoiPitch import InteractiveVoronoiPitch
import matplotlib.pyplot as plt
from FormationGenerator import FormationGenerator
import Configurations as CF

formationGenerator = FormationGenerator()

team_A = formationGenerator.generate_template_from_formation(formation_struct="5-3-2", team_side="left", mode="Midfield", shape="Flat")
team_B = formationGenerator.generate_template_from_formation(formation_struct="4-3-1-2", team_side="right", mode="Defending", shape="")

app = InteractiveVoronoiPitch(team_A, team_B)
plt.show()