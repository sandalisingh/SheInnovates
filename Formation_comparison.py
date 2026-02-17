from InteractiveVoronoiPitch import InteractiveVoronoiPitch
import matplotlib.pyplot as plt
from FormationGenerator import FormationGenerator
import Configurations as CF

formationGenerator = FormationGenerator()

team_A = formationGenerator.generate_template_from_formation(formation_struct=CF.Formation.F_5_3_2, team_side="left", mode=CF.Mode.MIDFIELD, shape=CF.Shape.FLAT)
team_B = formationGenerator.generate_template_from_formation(formation_struct=CF.Formation.F_4_3_1_2, team_side="right", mode=CF.Mode.DEFENDING, shape=CF.Shape.NA)

app = InteractiveVoronoiPitch(team_A, team_B)
plt.show()