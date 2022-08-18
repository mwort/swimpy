"""
Extended functionality related to the modelmanager browser plugin.
"""
from django.db import models
from django.conf import settings
from modelmanager.plugins.browser.database.models import Run
from swimpy.plot import plot_summary


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
            elif all([isinstance(i, Run) for i in runs]):
                return self.filter(pk__in=[i.pk for i in runs])
            else:
                raise TypeError(ermsg)
        else:
            raise TypeError(ermsg)
        return

    def to_frame(self, indicators=False, queryset=None, **filters):
        """Return runs table as pandas.DataFrame.

        Arguments
        ---------
        indicators : bool | list | str
            Add indicators to the DataFrame. If True, all will be expanded,
            otherwise just those in list or the one parsed as str.
        queryset : django.QuerySet, optional
            Convert specific queryset ignoring filters.
        filters :
            Apply any filter to the table.
        """
        import pandas as pd
        qs = queryset or self.filter(**filters)
        frame = pd.DataFrame.from_records(qs.values())
        frame.set_index('id', inplace=True)
        # return in correct column order
        cols = [f.name for f in self.model._meta.get_fields()]
        frame = frame[[c for c in cols if c in frame.columns]]

        if indicators:
            ind = [indicators] if type(indicators) == str else indicators
            for r in qs:
                for i in r.indicators.all():
                    if indicators is True or i.name in ind:
                        frame.loc[r.id, i.name] = i.value
        return frame

    def reset_ids(self):
        """Reset the ID counting to the last ID found in the model.

        Useful after many objects have been deleted.
        """
        from django.db import connection
        sql = "UPDATE SQLITE_SEQUENCE SET SEQ=%s WHERE NAME='browser_run';"
        maxid = self.latest('id').id if self.all().count() else 0
        with connection.cursor() as c:
            c.execute(sql, [maxid])
        return


class SwimRun(Run):
    class Meta:
        abstract = True

    objects = RunManager()

    # extra fields
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)
    run_time = models.DurationField(blank=True, null=True)

    @property
    def file_interfaces(self):
        """Attribute to let the Run object map files to plugins.
        """
        p = settings.PROJECT
        fi = {n: p.settings.properties[n] for n in p.output_files}
        return dict(files=fi)

    @property
    def plot_summary(self):
        return plot_summary(settings.PROJECT, self)
    # short alias
    plot = plot_summary
