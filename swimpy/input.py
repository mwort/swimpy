"""
SWIM input functionality.
"""
import pandas as pd


def basin_parameters(self, *getvalues, **setvalues):
    """
    Set or get any values from the .bsn file by variable name.
    """
    pat = 'input/*.bsn'
    if getvalues or setvalues:
        result = self.templates(templates=pat, *getvalues, **setvalues)
    # get all values if no args
    else:
        result = self.templates[pat].read_values()
    return result


def config_parameters(self, *getvalues, **setvalues):
    """
    Set or get any values from the .cod or swim.conf file by variable name.
    """
    pat = ['input/*.cod', 'swim.conf']
    if getvalues or setvalues:
        result = self.templates(templates=pat, *getvalues, **setvalues)
    else:  # get all values if no args
        result = self.templates[pat[0]].read_values()
        result.update(self.templates[pat[1]].read_values())
    return result


def subcatch_parameters(self, *getvalues, **setvalues):
    '''
    Read or write parameters in the subcatch.bsn file.

    Reading:
    --------
    (<param> or <stationID>): returns pd.Series
    (<list of param/stationID>): returns subset pa.DataFrame
    (): returns entire table as pd.DataFrame

    Writing:
    --------
    (<param>=value, <stationID>=<list-like>): Assign values or list to
        parameter column or stationID row. Is inserted if not existent.
    (<param>={<stationID>: value, ...}): Set individual values.
    (<pd.DataFrame>): Override entire table, DataFrame must have stationID
        index.
    '''
    filepath = self.subcatch_parameter_file
    # read subcatch.bsn
    bsn = pd.read_table(filepath, delim_whitespace=True)
    stn = 'stationID' if 'stationID' in bsn.columns else 'station'
    bsn.set_index(stn, inplace=True)
    is_df = len(getvalues) == 1 and isinstance(getvalues[0], pd.DataFrame)

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
        with open(filepath, 'w') as f:
            f.write(strtbl)
        return

    if getvalues:
        if all([k in bsn.index for k in getvalues]):
            return bsn.loc[getvalues]
        elif all([k in bsn.columns for k in getvalues]):
            ix = getvalues[0] if len(getvalues) == 1 else list(getvalues)
            return bsn[ix]
        else:
            raise KeyError('Cant find %s in either paramter or stations'
                           % getvalues)
    return bsn
