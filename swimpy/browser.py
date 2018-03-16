"""
Functionality related to the modelmanager browser plugin.
"""
from modelmanager.plugins import browser
from django.db import models
from django.conf import settings

from .utils import ProjectOrRunData


class SwimRun(browser.database.models.Run):
    class Meta:
        abstract = True
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)

    def __init__(self, *args, **kwargs):
        super(SwimRun, self).__init__(*args, **kwargs)
        # attach result propertyplugins
        for n, p in settings.PROJECT.settings.properties.items():
            if (hasattr(p, 'is_plugin') and
               ProjectOrRunData in p.plugin_class.__bases__):
                setattr(self.__class__, n, p)
        return
