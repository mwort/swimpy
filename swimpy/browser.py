"""
Extended functionality related to the modelmanager browser plugin.
"""
from django.db import models
from django.conf import settings
from modelmanager.plugins.browser.database.models import Run
from modelmanager.plugins.pandas import ProjectOrRunData


class RunManager(models.Manager):
    def get_runs(self, runs):
        """Transform a flexible runs argument into a QuerySet."""
        ermsg = ('The runs argument must be a run id (int), a Run '
                 'instance or an interable of these or a django QuerySet.')
        if isinstance(runs, Run):
            return self.filter(pk=runs.pk)
        elif type(runs) is int:
            return self.filter(pk=runs)
        elif isinstance(runs, models.query.QuerySet):
            return runs
        elif hasattr(runs, '__iter__'):
            if all([type(i) is int for i in runs]):
                return self.filter(pk__in=runs)
            elif all([type(i) is Run for i in runs]):
                return self.filter(pk__in=[i.pk for i in runs])
            else:
                raise TypeError(ermsg)
        else:
            raise TypeError(ermsg)
        return

    def as_frame(self, expand_indicators=False, **filters):
        import pandas as pd
        frame = pd.DataFrame.from_records(self.filter(**filters).values())
        frame.set_index('id', inplace=True)
        # return in correct column order
        cols = [f.name for f in self.model._meta.get_fields()]
        return frame[[c for c in cols if c in frame.columns]]


class SwimRun(Run):
    class Meta:
        abstract = True

    objects = RunManager()
    # class attribute to let the Run object know which plugins to associate
    # with what file.
    file_interfaces = {
      'resultfiles': {
        n: p for n, p in settings.PROJECT.settings.properties.items()
        if hasattr(p, 'plugin') and ProjectOrRunData in p.plugin.__bases__
        }}
    # extra fields
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)
