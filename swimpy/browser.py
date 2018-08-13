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

    def wait(self, nruns, interval=5, timeout={'hours': 12}, **filters):
        """Wait until nruns are found (with the filters) and return them.

        Arguments
        ---------
        nruns : int
            Number of runs to wait for.
        interval : int seconds
            Polling interval in seconds.
        timeout : dict
            Raise RuntimeError after timeout is elapsed. Parse any keyword
            to datetime.timedelta, e.g. hours, days, minutes, seconds.
        **filters :
            Any run attribute filter to query the runs table with.

        Returns
        -------
        QuerySet
        """
        from sys import stdout
        from time import sleep
        import datetime as dt
        st = dt.datetime.now()
        to = dt.timedelta(**timeout)
        fltstr = ', '.join('%s=%r' % kv for kv in filters.items()) or 'all'
        msg = "\rWaiting for %s runs (%s) for %s hh:mm:ss"
        ndone = 0
        while ndone < nruns:
            runs = self.filter(**filters) if filters else self.all()
            ndone = runs.count()
            et = dt.datetime.now()-st
            if ndone < nruns:
                if et > to:
                    em = '%s runs not found within %s hh:mm:ss' % (nruns, to)
                    raise RuntimeError(em)
                stdout.write(msg % (nruns-ndone, fltstr, et))
                stdout.flush()
                sleep(interval)
        stdout.write("\n")
        return runs


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
