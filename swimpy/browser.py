"""
Functionality related to the modelmanager browser plugin.
"""
from django.db import models
from django.conf import settings
from modelmanager.plugins.browser.database.models import Run

from .utils import ProjectOrRunData


class SwimRun(Run):
    class Meta:
        abstract = True

    file_interfaces = {
      'resultfiles': {
        n: p for n, p in settings.PROJECT.settings.properties.items()
        if hasattr(p, 'plugin') and ProjectOrRunData in p.plugin.__bases__
        }}

    # extra fields
    start = models.DateField('Start date', null=True)
    end = models.DateField('End date', null=True)
