#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `swimpy` package."""

import os
import unittest

from swimpy import Project


SWIM_TEST_PROJECT = 'swim/project/'

if not os.path.exists(SWIM_TEST_PROJECT):
    raise IOError('The SWIM test project is not located at: %s'
                  % SWIM_TEST_PROJECT)


class TestFunctions(unittest.TestCase):
    """Tests for `swimpy` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""
