import os.path as osp
import os
import unittest

import pandas as pd

import swimpy

BENCHMARK_PATH = 'blankenstein_output_sums'
PROJECTDIR = 'project'


class OutputSums(unittest.TestCase):
    precision = 5

    @classmethod
    def setUpClass(cls):
        cls.project = swimpy.Project(PROJECTDIR)

    def test_sums(self):
        for i in self.project.output_interfaces:
            df = getattr(self.project, i)
            if df.path:
                bpth = _benchmark_path(df.path)
                benchmark = pd.read_pickle(bpth)
                # check each column sum
                for n, c in df.sum().items():
                    b = benchmark[n]
                    # deviation for reporting
                    di = '%s%%' % (((c/b)-1)*100) if b else b-c
                    with self.subTest(path=df.path, column=n, deviation=di):
                        self.assertAlmostEqual(c, benchmark[n], self.precision)


def _benchmark_path(interface_path):
    odir = osp.join(PROJECTDIR, 'output')
    path = osp.join(BENCHMARK_PATH, osp.relpath(interface_path, odir))+'.pd'
    return path


def create_benchmark():
    project = swimpy.Project(PROJECTDIR)
    for o in project.output_interfaces:
        df = getattr(project, o)
        if df.path:
            path = _benchmark_path(df.path)
            os.makedirs(osp.dirname(path), exist_ok=True)
            print('Writing %s' % path)
            df.sum().to_pickle(path)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare SWIM output data columnwise to benchmark data.')
    parser.add_argument('-b', action='store_true',
                        help='Create (overwrite) benchmark data and exit.')
    args = parser.parse_args()
    if args.b:
        create_benchmark()
    else:
        unittest.main()
