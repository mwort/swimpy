"""Pytest configuration, shared fixtures and unittest compatibility shim.

The ``UnitTestAssertMixin`` provides assert helpers used by the unchanged
mixin classes in ``swimpy/tests/``.  ``ProjectTestCase`` exposes those helpers
plus the ``swim_project`` fixture so all test classes share a single project
instance across the session.
"""
# test_settings.py is a swimpy project settings file, not a test module
collect_ignore = ["test_settings.py"]

import errno
import os
import os.path as osp
import shutil

import pytest
import swimpy

# ---------------------------------------------------------------------------
# paths (mirroring the constants that used to live in tests.py)
# ---------------------------------------------------------------------------
SWIM_REPO = osp.join(osp.dirname(__file__), '..', 'dependencies', 'swim')
SWIM_TEST_PROJECT = osp.join(osp.dirname(__file__), 'project')
SWIM_REPO_PROJECT = osp.join(SWIM_REPO, 'project')
TEST_GRASSDB = osp.join(osp.dirname(__file__), 'grassdb')
MSWIM_GRASSDB = osp.join(osp.dirname(__file__), '..', 'dependencies',
                         'm.swim', 'test', 'grassdb')
TEST_SETTINGS = osp.join(osp.dirname(__file__), 'test_settings.py')


# ---------------------------------------------------------------------------
# NFS-safe rmtree
# ---------------------------------------------------------------------------
def _rmtree_nfs(path):
    """shutil.rmtree that silently ignores NFS-busy (.nfs*) files."""
    def onexc(func, fpath, exc):
        if isinstance(exc, OSError) and exc.errno in (errno.EBUSY,
                                                       errno.ENOTEMPTY):
            pass
        else:
            raise exc
    if osp.exists(path):
        shutil.rmtree(path, onexc=onexc)


# ---------------------------------------------------------------------------
# unittest compatibility shim
# ---------------------------------------------------------------------------
class UnitTestAssertMixin:
    """Thin wrappers around plain ``assert`` so the unchanged swimpy/tests/*
    mixin methods (which call ``self.assertEqual`` etc.) keep working without
    inheriting from ``unittest.TestCase``."""

    def assertEqual(self, first, second, msg=None):
        assert first == second, msg or f"{first!r} != {second!r}"

    def assertNotEqual(self, first, second, msg=None):
        assert first != second, msg or f"{first!r} == {second!r}"

    def assertTrue(self, expr, msg=None):
        assert expr, msg or f"Expected truthy, got {expr!r}"

    def assertFalse(self, expr, msg=None):
        assert not expr, msg or f"Expected falsy, got {expr!r}"

    def assertIn(self, member, container, msg=None):
        assert member in container, msg or f"{member!r} not in {container!r}"

    def assertNotIn(self, member, container, msg=None):
        assert member not in container, \
            msg or f"{member!r} unexpectedly in {container!r}"

    def assertIsInstance(self, obj, cls, msg=None):
        assert isinstance(obj, cls), \
            msg or f"{obj!r} is not an instance of {cls!r}"

    def assertIsNot(self, first, second, msg=None):
        assert first is not second, msg or f"{first!r} is {second!r}"

    def assertIsNone(self, obj, msg=None):
        assert obj is None, msg or f"{obj!r} is not None"

    def assertIsNotNone(self, obj, msg=None):
        assert obj is not None, msg or "Unexpectedly None"

    def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        if delta is not None:
            assert abs(first - second) <= delta, \
                msg or f"|{first} - {second}| > {delta}"
        else:
            assert round(abs(first - second), places) == 0, \
                msg or f"{first!r} != {second!r} within {places} decimal places"

    def assertLess(self, first, second, msg=None):
        assert first < second, msg or f"{first!r} >= {second!r}"

    def assertGreater(self, first, second, msg=None):
        assert first > second, msg or f"{first!r} <= {second!r}"

    def assertListEqual(self, first, second, msg=None):
        assert first == second, msg or f"{first!r} != {second!r}"

    def assertRaises(self, exc, *args, **kwargs):
        import contextlib
        if not args:
            return pytest.raises(exc)
        callable_, *call_args = args
        with pytest.raises(exc):
            callable_(*call_args, **kwargs)


# ---------------------------------------------------------------------------
# Shared base class (imported by all test modules)
# ---------------------------------------------------------------------------
class ProjectTestCase(UnitTestAssertMixin):
    """Base class for all project-level tests.

    Subclasses receive the class-scoped ``swim_project`` fixture via
    ``autouse=True`` (injected by ``conftest``).  The project is available as
    ``self.project``.  Each test class gets a fresh copy of the project,
    mirroring the original ``setUpClass`` behaviour.
    """
    project = None  # set by the autouse fixture below


# ---------------------------------------------------------------------------
# Class-scoped project fixture (mirrors the original setUpClass behaviour)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def swim_project():
    """Set up a fresh Blankenstein test project once per test class."""
    _rmtree_nfs(SWIM_TEST_PROJECT)
    _rmtree_nfs(TEST_GRASSDB)
    shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT, dirs_exist_ok=True)
    if osp.exists(MSWIM_GRASSDB):
        shutil.copytree(MSWIM_GRASSDB, TEST_GRASSDB, dirs_exist_ok=True)
    p = swimpy.project.setup(SWIM_TEST_PROJECT)
    shutil.copy(TEST_SETTINGS, p.settings.file)
    project = swimpy.Project(SWIM_TEST_PROJECT)
    project.browser.project.settings.load()
    yield project
    project.browser.settings.unset()
    _rmtree_nfs(project.projectdir)
    _rmtree_nfs(TEST_GRASSDB)


# ---------------------------------------------------------------------------
# Autouse fixture that injects swim_project into every ProjectTestCase instance
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _inject_project(request):
    """Inject the session project into ProjectTestCase instances."""
    if isinstance(request.instance, ProjectTestCase):
        request.instance.project = request.getfixturevalue('swim_project')
