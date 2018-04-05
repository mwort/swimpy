"""
Functionality related to the modelmanager browser plugin.
"""
from modelmanager.plugins import browser
from django.db import models
from django.conf import settings


class SwimRun(browser.database.models.Run):
    class Meta:
        abstract = True
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)

    def __init__(self, *args, **kwargs):
        super(SwimRun, self).__init__(*args, **kwargs)
        # attach result propertyplugins
        for n, p in settings.PROJECT.output.interfaces.items():
            setattr(self.__class__, n, p)
        return
