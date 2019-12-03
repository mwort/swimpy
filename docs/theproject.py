import swimpy
from modelmanager.project import ProjectDoesNotExist

FILE = 'theproject.rst'

TOP = """
The Project
===========

This is a list of registered attributes and methods of the
:class:`~swimpy.project.Project` instance (with the
:mod:`swimpy.defaultsettings`). Attributes dynamically added by
plugins are not shown here.

"""

try:
    p = swimpy.Project('../tests/project')
except ProjectDoesNotExist:
    raise IOError('Make sure test/project exists.')

l = ['\nAttributes\n----------\n']
# defaultsettings variables
for v in sorted(p.settings.variables.keys()):
    l.append(u"- :attr:`~swimpy.defaultsettings.%s`" % v)

l.append('\nFunctions\n---------\n')
# project functions
for f, o in sorted(p.settings.functions.items()):
    if len(f.split('.')) == 1 and f not in p.settings.plugins:
        l.append(u"- :meth:`~swimpy.project.Project.%s`" % f)

l.append('\nPlugins\n-------\n')
# plugins
for n, o in sorted(p.settings.plugins.items()):
    if len(n.split('.')) == 1:
        path = o.__module__+'.'+o.__name__
        l.append(u"- :class:`%s <%s>`" % (n, path))

with open(FILE, 'w') as f:
    f.write(TOP)
    f.write('\n'.join(l))

# clean
p.browser.settings.unset()
