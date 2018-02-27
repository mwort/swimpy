# -*- coding: utf-8 -*-

"""Unit test package for swimpy."""
import glob
import os.path as osp
import unittest
import inspect
import types

from modelmanager.utils import load_module_path


class Tests:
    def __init__(self, project):
        self.project = project
        # add all test cases (mixin classes)
        testpaths = glob.glob(osp.join(osp.dirname(__file__), 'test_*.py'))
        for modpath in testpaths:
            modulename = osp.splitext(osp.basename(modpath))[0]
            testmodule = load_module_path(modulename, modpath)
            classes = [getattr(testmodule, i) for i in dir(testmodule)
                       if inspect.isclass(getattr(testmodule, i))]
            for Tc in classes:
                test_function = self.create_test_function(Tc)
                # attach function to Tests class
                method = types.MethodType(test_function, self)
                setattr(self, Tc.__name__.lower(), method)
        return

    def create_test_function(self, testcaseclass):
        PROJECT = self.project

        class TestCase(unittest.TestCase, testcaseclass):
            project = PROJECT

        def test_function(self):
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestCase)
            return unittest.TextTestRunner().run(suite)
        return test_function
