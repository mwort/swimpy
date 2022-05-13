
from swimpy import Project, dashboard


swim = Project("tests/project", stations=None)

print(swim.config_parameters.start_date)
dashboard.App(swim).start()