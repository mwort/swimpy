# -*- coding: utf-8 -*-

"""Unit test package for any swimpy project."""
import glob
import os.path as osp
import unittest
import inspect
import types

from modelmanager.utils import load_module_path


class test(object):
    """The swimpy test project plugin.

    The plugin makes project-unspecifc tests available to the project instance
    through the following syntax::

        project.test.<testcaseclass>()
        project.test.all()
        project.test()  # same as all

    Test case classes maybe defined in any submodule of the tests package and
    should be named ``test_*.py``.
    """

    def __init__(self, project):
        self.project = project
        self.test_methods = {}
        # add all test cases (mixin classes)
        testpaths = glob.glob(osp.join(osp.dirname(__file__), 'test_*.py'))
        if project.resourcedir:
            testpaths += glob.glob(osp.join(project.resourcedir, 'test_*.py'))
        for modpath in testpaths:
            testmodule = load_module_path(modpath)
            classes = self._get_classes(testmodule)
            for Tc in classes:
                self._add_test_method(Tc)
        return

    def _get_classes(self, obj):
        vrs = [v for v in dir(obj) if not v.startswith('_')]
        objts = [getattr(obj, v) for v in vrs]
        return [o for o in objts if inspect.isclass(o)]

    def _add_test_method(self, testcaseclass):
        PROJECT = self.project
        name = testcaseclass.__name__.lower()

        class TestCase(unittest.TestCase, testcaseclass):
            project = PROJECT

        def test_function(self, test='all', list=False):
            """
            Arguments
            ---------
            test : string | 'all'
                Only run specific test method (without test_ prefix)
                if not all.
            list : bool
                List all test methods with doc strings and exit.
            """
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestCase)
            if list:
                for e in suite:
                    print(e._testMethodName[len('test_'):] + ' : ', end='')
                    print(e._testMethodDoc or '')
                return
            if test == "all":
                return unittest.TextTestRunner().run(suite)
            else:
                methods = {e._testMethodName[len('test_'):]: e for e in suite}
                assert test in methods, "Not a valid test name: %s" % test
                return unittest.TextTestRunner().run(methods[test])

        test_function.__name__ = name
        test_function.__doc__ = ((testcaseclass.__doc__ or 'SWIM tests\n') +
                                 test_function.__doc__)
        test_method = types.MethodType(test_function, self)
        # attach function to Tests class
        setattr(self, name, test_method)
        self.test_methods[name] = test_method
        testadd = self.__class__.__name__+'.'+name
        self.project.settings.register_function(test_method, testadd)
        return test_method

    def all(self):
        for n, m in self.test_methods.items():
            m()
        return

    def __call__(self):
        self.all()
        return
