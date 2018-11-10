import swimpy
import shutil

FILE = 'theproject.rst'

TOP = """
The Project
===========

This is a list of registered attributes and methods of the ``Project`` instance
(with the :mod:`swimpy.defaultsettings`). Attributes dynamically added by
plugins are not shown here.

- :class:`~swimpy.project.Project`

"""

p = swimpy.project.setup('../tests/project', name='blank',
                         gitrepo='../dependencies/swim')

l = []

# defaultsettings variables
for v in sorted(p.settings.variables.keys()):
    l.append(u"  - :attr:`~swimpy.defaultsettings.%s`" % v)

# project functions
for f in sorted(p.settings.functions.keys()):
    if len(f.split('.')) == 1 and f not in p.settings.plugins:
        l.append(u"  - :meth:`~swimpy.project.Project.%s`" % f)

# plugins
for n, o in sorted(p.settings.plugins.items()):
    if len(n.split('.')) == 1:
        path = o.__module__+'.'+o.__name__
        if n in p.settings.properties:
            path += '.plugin'
        l.append(u"  - :class:`%s <%s>`" % (n, path))

with open(FILE, 'w') as f:
    f.write(TOP)
    f.write('\n'.join(l))

# clean
shutil.rmtree(p.resourcedir)
