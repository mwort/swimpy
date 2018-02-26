"""
Your project-specific database tables (django calls them 'models').
The modelmanager.plugins.browser.models module already defines several
'abstract models' that can be used as extensible templates as has been done
below.

Here is an example:

```
from django.db import models
from modelmanager.plugins import browser

# the run table with an additional type field
class Run(metamodels.Run):
    type = models.FloatField('xyzmetric')

# or an entirely new model using the Django Model class
class Output_XYZ(models.Model):
    name = models.CharField(max_length=100)
    time = models.DateTimeField('Time', auto_now_add=True)
    type = models.FloatField('xyzmetric')
    run = models.OneToOneField(Run, on_delete=models.CASCADE)
```

Refer to https://docs.djangoproject.com/en/dev/ref/models/fields/ for fields
and documentation.
"""

from django.db import models
from modelmanager.plugins.browser.database import models as mmmodels


class Run(mmmodels.Run):
    # columns shown in browser table, can also be method names!!!
    show_columns = ['id', 'time', 'tags', 'notes']
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)
    pass


class Parameter(mmmodels.TaggedValue):
    pass


class ResultIndicator(mmmodels.TaggedValue):
    pass


class ResultFile(mmmodels.File):
    pass
