# -*- coding: utf-8 -*-

"""
The main project module.
"""
import os
import os.path as osp
import shutil

import pandas as pa

import modelmanager


class Project(modelmanager.Project):

    def basin_parameters(self, *getvalues, **setvalues):
        """
        Set or get any values from the .bsn file by variable name.
        """
        tmplt = self.templates['input/*.bsn']
        if len(setvalues) > 0:
            result = tmplt.write_values(**setvalues)
        else:
            result = tmplt.read_values(*getvalues)
        return result

    def config_parameters(self, *getvalues, **setvalues):
        """
        Set or get any values from the .cod file by variable name.
        """
        tmplt = self.templates['input/*.cod']
        if len(setvalues) > 0:
            result = tmplt.write_values(**setvalues)
        else:
            result = tmplt.read_values(*getvalues)
        return result

    def subcatch_parameters(self, *getvalues, **setvalues):
        '''
        Read or write parameters in the subcatch.bsn file.

        # Reading
        subcatch_parameters(<param> or <stationID>) -> pa.Series
        subcatch_parameters(<list of param/stationID>) -> subset pa.DataFrame
        subcatch_parameters() -> pa.DataFrame of entire table

        # Writing
        # assign values or list to parameter column / stationID row (maybe new)
        subcatch_parameters(<param>=value, <stationID>=<list-like>)
        # set individual values
        subcatch_parameters(<param>={<stationID>: value, ...})
        # override entire table, DataFrame must have stationID index
        subcatch_parameters(<pa.DataFrame>)
        '''
        filepath = self.subcatch_paramter_file
        # read subcatch.bsn
        bsn = pa.read_table(filepath, delim_whitespace=True)
        stn = 'stationID' if 'stationID' in bsn.columns else 'station'
        bsn.set_index(stn, inplace=True)
        is_df = len(getvalues) == 1 and isinstance(getvalues[0], pa.DataFrame)

        if setvalues or is_df:
            if is_df:
                bsn = getvalues[0]
            for k, v in setvalues.items():
                ix = slice(None)
                if type(v) == dict:
                    ix, v = zip(*v.items())
                if k in bsn.columns:
                    bsn.loc[ix, k] = v
                else:
                    bsn.loc[k, ix] = v
            # write table again
            bsn['stationID'] = bsn.index
            strtbl = bsn.to_string(index=False, index_names=False)
            open(filepath, 'w').write(strtbl)
            return

        if getvalues:
            if all([k in bsn.index for k in getvalues]):
                return bsn.loc[k]
            elif all([k in bsn.columns for k in getvalues]):
                ix = getvalues[0] if len(getvalues) == 1 else list(getvalues)
                return bsn[ix]
            else:
                raise KeyError('Cant find %s in either paramter or stations'
                               % getvalues)
        return bsn


def setup(projectdir='.', resourcedir='swimpy'):
    """
    Setup a swimpy project.
    """
    mmproject = modelmanager.project.setup(projectdir, resourcedir)
    # swim specific customisation of resourcedir
    defaultresourcedir = osp.join(osp.dirname(__file__), 'resources')
    for path, dirs, files in os.walk(defaultresourcedir, topdown=True):
        relpath = os.path.relpath(path, defaultresourcedir)
        for f in files:
            dst = osp.join(mmproject.resourcedir, relpath, f)
            shutil.copy(osp.join(path, f), dst)
        for d in dirs:
            dst = osp.join(mmproject.resourcedir, relpath, d)
            if not osp.exists(dst):
                os.mkdir(dst)
    # FIXME rename templates with project name in filename
    for fp in ['cod', 'bsn']:
        ppn = modelmanager.utils.get_paths_pattern('input/*.' + fp, projectdir)
        tp = osp.join(mmproject.resourcedir, 'templates')
        os.rename(osp.join(tp, 'input/%s.txt' % fp), osp.join(tp, ppn[0]))
    # load as a swim project
    project = Project(projectdir)
    return project
