"""A collection of Evolutionary Multiobjective Optimization Algorithms.

The algorithm implementation is provided by the *evoalgos* package. This module
implements them for SWIM.

The algorithms are described here:
https://ls11-www.cs.tu-dortmund.de/people/swessing/evoalgos/doc/algo.html

To run the algorithm either via SLURM (sbatch jobfiles) or python native
multiprocessing, import any of the algorithms in your settings.py file and
execute them as a project method. Refer to the docs of the __call__ method for
help.
"""
from __future__ import print_function, absolute_import
import random
import os.path as osp
import warnings
import collections
import datetime as dt


import numpy as np
import pandas as pd
from modelmanager.settings import parse_settings
from modelmanager.utils import propertyplugin
from modelmanager.plugins.pandas import ProjectOrRunData

from swimpy.plot import plot_objective_scatter, plot_function, plt

try:
    from evoalgos import algo
    from evoalgos.individual import SBXIndividual
    from optproblems import Problem
except ImportError:
    raise ImportError('The %s module requires the evoalgos ' % __name__ +
                      'package. Try `pip install evoalgos`')


class _EvoalgosSwimProblem(Problem):
    '''The evoalgos.Problem to be run with any of the algorithms provided.'''
    #: Default algorithm
    algorithm = None
    #: Value to use if nan is returned by objective function
    nanval = 2e31
    #: Attributes of individuals written to output
    output_attribute_columns = ['generation', 'id_number',
                                'clone', 'birthgeneration']
    #: factor to multiply the average runtime for job time out estimation
    time_safty_factor = 3

    plugin = ['__call__']

    def __init__(self, project):
        self.project = project
        self.algorithm = self.algorithm or self.__class__.__name__
        msg = ('Either set the algorithm class attribute or name the subclass '
               'after a valid evoalgos algorithm.')
        assert hasattr(algo, self.algorithm), msg
        self.clones = []
        self.__call__.__func__.__doc__ = getattr(algo, self.algorithm).__doc__
        # output interface
        # TODO: does not work, i.e. output object has no
        # optimization_populations attribute (worked in old swimpy)
        self.project.settings(propertyplugin(optimization_populations))
        return

    @parse_settings
    def __call__(self, parameters=None, objectives=None, population_size=10,
                 max_generations=10, output=None, restart=False, test=None, prefix=None,
                 keep_clones=False, parameter_setter="config_parameters", **kwargs):
        """Run the optimisation algorithm.

        Arguments
        ---------
        parameters : dict of {'name': (lower, upper)}
            Dictionary of parameter names and upper and lower boundaries as
            values (by default) or any other values to determine their initial
            value if the ``create_start_population`` method is overridden.
            This is only a keyword parameter to allow setting it via the
            settings file.
        objectives : list of strings or dict of strings
            Performance indicators passed to project.run. Only strings are
            accepted. Pass a dict to redefine names.
        population_size : int
            Population size the algorithm is run with. By default this is also
            the ``num_offspring`` argument to the algorithm.
        max_generation : int
            Maximum number of iterations the algorithm is run with.
        output : string path, optional
            Output .csv file with saved parameters and performances for each
            generation and individual (default:
            [<prefix>_]<algorithm>_populations.csv).
        test : bool | None
            Only run tests before running algorithm if True, dont run them if
            False and run them before running the algorithm if None (default).
        prefix : str, optional
            A prefix to use for run tags and project clones. Defaults to
            algorithm name.
        restart : bool
            Attempt to restart the algorithm if output exists.
        keep_clones : bool
            Do not remove the project clones when completed and run the final
            population in them.
        parameter_setter : str
            Project method that receives the parameter dict to set it. E.g. this could
            be "catchment" for catchment specific calibrations.
            Defaults to config_parameters.
        kwargs :
            Any overriding parameter parsed to the algorithm (for details see
            the algorithm descriptions at:
            https://ls11-www.cs.tu-dortmund.de/people/swessing/evoalgos/doc/algo.html)

        Returns
        -------
        <browser.Run> :
            A run instance with the mean objective values as indicators,
            the mean parameter values of the last generation and the output
            file attached.
        """
        st = dt.datetime.now()
        assert type(parameters) == dict
        self.parameters = collections.OrderedDict(sorted(parameters.items()))
        self.objectives, self.indicators = self._parse_objectives(objectives)
        self.prefix = prefix or self.algorithm
        do = prefix+'_'+self.algorithm if prefix else self.algorithm
        defout = osp.join(self.project.projectdir, do+'_populations.csv')
        self.output = output or defout
        self.restart = restart and osp.exists(self.output)
        self.parameter_setter = parameter_setter
        assert hasattr(self.project, self.parameter_setter)
        # init problem
        Problem.__init__(self, lambda dummy: dummy,
                         num_objectives=len(objectives),
                         name=self.__class__.__name__)
        # set defaults
        self.population_size = kwargs.setdefault('population_size',
                                                 population_size)
        self.max_generations = kwargs.setdefault("max_generations",
                                                 max_generations)
        self.num_offspring = kwargs.setdefault("num_offspring",
                                               self.population_size)
        stpop = self.restart_population() if self.restart \
            else self.create_start_population()
        self.start_population = kwargs.setdefault('start_population', stpop)
        # run tests if test is True (and exit) or None (and continue)
        self.evaltimes = []
        self.max_run_time = None
        if test is not False and not self.restart:
            print('Testing single run...')
            tst = dt.datetime.now()
            self.run_tests()
            self.evaltimes.append(dt.datetime.now()-tst)
            self.max_run_time = self.evaltimes[0]
        if test is True:
            return
        # initialise algorithm
        self.ea = getattr(algo, self.algorithm)(self, **kwargs)
        # attach observer function
        self.ea.attach(self.observe_population)
        # set generation if restart
        if self.restart:
            self.ea.generation = self.read_populations().index.levels[0][-1]
            self.ea.remaining_generations = max_generations-self.ea.generation
            print('Restarting from generation %i' % self.ea.generation)
        else:
            # write initial population to file
            self.observe_population(self.ea, initial=True)
        # create clones
        self.clones = self._create_clones()
        # process
        self.ea.run()
        # run final population again or remove
        if keep_clones:
            self.batch_evaluate(self.ea.population)
        else:
            for c in self.clones:
                self.project.clone[c].remove()
        # reset runIDs
        self.project.browser.runs.reset_ids()
        run = self._save_run()
        print('Elapsed time: %s hh:mm:ss' % (dt.datetime.now()-st))
        return run

    def _parse_objectives(self, objectives):
        assert type(objectives) in [list, dict]
        if type(objectives) == dict:
            assert all([type(v) == str for v in objectives.values()])
            o, i = zip(*sorted(objectives.items()))
            o, i = list(o), list(i)
        else:
            o = sorted(objectives)
            i = sorted(objectives)
        assert all([type(k) == str for k in objectives])
        return o, i

    def _save_run(self):
        # insert run
        pops = self.read_populations()
        indi = {pops.indicators[n]: {'mean_final_population': v}
                for n, v in pops.lastgen[pops.objectives].mean().items()}
        pars = [{'name': n, 'value': v, "tags": 'mean_final_population'}
                for n, v in pops.lastgen[pops.parameters].mean().items()]
        # unique tags
        tags = ' '.join(set([self.prefix, self.algorithm]))
        notes = ("population_size=%s, max_generations=%s"
                 % (self.population_size, self.max_generations))
        run = self.project.save_run(indicators=indi, files={tags: pops},
                                    parameters=pars, tags=tags, notes=notes)
        return run

    def batch_evaluate(self, individuals):
        """Evaluate/run a batch of individuals/projects in parallel.

        Arguments
        ---------
        individuals : list of <evoalgos.Individual>s
        """
        # only run if there is anything to evaluate, not checked in evoalgos
        if not individuals:
            return
        st = dt.datetime.now()
        # start swim runs
        pnames = self.parameters.keys()
        for i, ind in enumerate(individuals):
            clone = self.project.clone[self.clones[i]]
            ind.clonename = clone.clonename
            self.set_parameters(clone, dict(zip(pnames, ind.phenome)))
            del clone

        runs = self.batch_run()
        objective_values = self.retrieve_objectives(runs)
        mrt = max(runs.values_list('run_time', flat=True))
        self.max_run_time = max(self.max_run_time or dt.timedelta(0), mrt)
        # delte all runs again
        runs.delete()
        # assign values to individuals
        for i in individuals:
            i.objective_values = objective_values[i.clonename]
        self.evaltimes += [dt.datetime.now()-st]
        return

    def batch_run(self):
        mrt = self.max_run_time
        rt = (int(round(mrt.total_seconds()*self.time_safty_factor/60. + 0.5))
              if mrt else None)
        # process clones and wait for runs
        runs = self.project.cluster.run_parallel(
            self.clones, time=rt, indicators=self.indicators, parameters=False)
        return runs

    def _create_clones(self):
        cn = self.prefix+('_%'+'0%0ii' % len(str(self.population_size-1)))
        cnames = []
        for i in range(self.population_size):
            name = cn % i
            self.project.clone(name)
            cnames.append(name)
        return cnames

    def retrieve_objectives(self, runs):
        """Get objective values from the browser database.

        Arguments
        ---------
        runs : browser.Run QuerySet
        """
        obvals = {}
        for r in runs:
            t, clonename = r.tags.split()
            ri = r.indicators.all()
            vals = []
            for i in self.indicators:
                rv = ri.filter(name=i)
                if len(rv) == 1:
                    vals += [float(rv[0].value)]
                else:
                    print(i+' for '+clonename+' returned %s.' % rv +
                          'Will set it to %s.' % self.nanval)
                    vals += [self.nanval]
            obvals[clonename] = vals
        return obvals

    def run_tests(self, quiet=False):
        """Execute a series of tests before running the algorithm.

        Tests:

        - creates clone
        - calls self.set_parameters
        - runs clone
        - checks if returned run contains the same number of indicators
          as self.indicators
        - checks if retrived_objectives return same number of values as
          self.objectives
        """
        try:
            clone = self.project.clone(self.prefix+'__test', fresh=True)
            params0 = dict(zip(self.parameters.keys(),
                               self.start_population[0].genome))
            self.set_parameters(clone, params0)
            run = clone.run(indicators=self.indicators, quiet=quiet,
                            tags='run_test '+clone.clonename, parameters=False)
            assert run.indicators.all().count() == len(self.indicators)
            runqset = clone.browser.runs.filter(tags__contains=clone.clonename)
            assert runqset.count() == 1
            obj_vals = self.retrieve_objectives(runqset)[clone.clonename]
            assert len(obj_vals) == len(self.objectives)
            obj_str = ['%s=%s' % (k, v)
                       for k, v in zip(self.objectives, obj_vals)]
            print('Test objective values:\n%s' % ('\n'.join(obj_str)))
        except Exception:
            raise
        finally:
            try:
                clone.remove()
                run.delete()
            except NameError:
                pass
        return

    def create_start_population(self):
        """Create the initial population out of the parameter boundaries.

        Returns
        -------
        list of evoalgos.individual.SBXIndividual s
        """
        population = []
        lo, up = zip(*self.parameters.values())
        for _ in range(self.population_size):
            parameters = [random.uniform(l, u) for l, u in zip(lo, up)]
            kw = {"min_bounds": lo, "max_bounds": up, 'clonename': None}
            population.append(self.create_individual(parameters, **kw))
        return population

    def restart_population(self):
        """Read output and return the last generation as population."""
        opop = self.read_populations()
        assert set(opop.parameters) == set(self.parameters), \
            'Restart parameters dont match parsed.'
        params = self.parameters.keys()  # OrderedDict
        assert len(opop.last_generation) == self.population_size
        # using the old parameter ranges from file
        lo, up = zip(*[opop.parameter_ranges[i] for i in params])
        population = []
        for id, prs in opop.last_generation.iterrows():
            parameters = prs[params].tolist()
            kw = {"min_bounds": lo, "max_bounds": up, 'id': id,
                  'clonename': prs['clone'],
                  'objective_values': prs[self.objectives].tolist(),
                  'date_of_birth': prs['birthgeneration']}
            population.append(self.create_individual(parameters, **kw))
        return population

    def create_individual(self, parameters, **kwargs):
        """Create a single evoalgos individual with parameters and bounds."""
        indiv = SBXIndividual(genome=parameters)
        for k, v in kwargs.items():
            setattr(indiv, k, v)
        return indiv

    def set_parameters(self, clonedproject, parameters):
        """Default parameter setting in method for convenient overriding.

        Arguments
        ---------
        clonedproject : <swimpy.Project>
            Instance of the cloned project.
        parameters : dict
            Parameters to set.
        """
        getattr(clonedproject, self.parameter_setter, "config_parameters")(**parameters)
        return

    def observe_population(self, ea, initial=False):
        """Evoalgos function to write the population to the output file.

        The function only takes the algorithm instance as first argument and
        doesnt return anything.
        """
        # columns to write out
        pars = ['parameter:%s:%r' % kv for kv in self.parameters.items()]
        objs = ['objective:%s:%s' % oi
                for oi in zip(self.objectives, self.indicators)]
        cols = self.output_attribute_columns + objs + pars
        # collect population info
        popinfolist = []
        for i in self.ea.population:
            iline = [self.ea.generation+(0 if initial else 1), i.id_number,
                     i.clonename, i.date_of_birth]
            iline += list(i.objective_values or [None]*len(self.objectives))
            iline += list(i.genome)
            popinfolist.append(iline)
        # make dataframe and write out
        popframe = pd.DataFrame(popinfolist, columns=cols)
        with open(self.output, 'w' if initial else 'a') as f:
            popframe.to_csv(f, header=initial, index=False)
        # report stats
        if not initial:
            obj_stats = popframe[objs].describe().T[['50%', 'min']]
            mt = self.mean_generation_time()
            rt = self.max_run_time
            mg = mt*(self.max_generations-self.ea.generation-1)
            ovstr = ['%s: %3.6f %3.6f' % (o, i[0], i[1])
                     for o, i in zip(self.objectives, obj_stats.values)]
            msg = ('Generation %s completed in %s, mean generation time %s, ' +
                   'max run time %s, max_generations in ~%s hh:mm:ss\n' +
                   'Objectives (median, min):\n' + '\n'.join(ovstr))
            print(msg % (self.ea.generation+1, self.evaltimes[-1], mt, rt, mg))
        return

    def mean_generation_time(self):
        """Calculate the average time taken for a generation to finish.

        Returns
        -------
        datetime.timedelta
        """
        # giving dt.timedelta(0) as the start value makes sum work on tds
        nt = max(len(self.evaltimes), 1)
        return sum(self.evaltimes, dt.timedelta(0)) / nt

    def read_populations(self, filepath=None):
        assert filepath or self.output
        path = filepath or self.output
        return optimization_populations(self.project).from_path(path)


class SMSEMOA(_EvoalgosSwimProblem):
    pass


class CommaEA(_EvoalgosSwimProblem):
    pass


class CMSAES(_EvoalgosSwimProblem):
    pass


class NSGA2b(_EvoalgosSwimProblem):
    pass


class optimization_populations(ProjectOrRunData):
    """Dataframe to handle successive populations from a genetic optimisation.

    Note
    ----
    Class of propertyplugin that will be added to settings by the algorithm
    plugin. It enables reading population files via::

        project.optimization_populations.from_path(path)
        run.optimization_populations  # if the run was saved with algorithm
        project.algorithm.read_populations  # with_EvoalgosSwimProblem subclass

    """

    path = None
    index_col = (0, 1)
    _metadata = ['parameters', 'parameter_ranges', 'objectives', 'indicators']
    plugin = ['plot_generation_objectives', 'plot_objective_scatter',
              'plot_parameter_distribution', 'best_tradeoff', "from_path"]

    def from_csv(self, path, **readkw):
        """Read csv file and interpret objectives and parameter ranges from
        columns.
        """
        df = pd.read_csv(path, index_col=self.index_col, **readkw)
        # interpret columns
        cols = []
        self.parameters = []
        self.parameter_ranges = {}
        self.objectives = []
        self.indicators = {}
        for c in df.columns:
            pr = c.split(':')
            if len(pr) >= 3:
                c = pr[1]
                if pr[0] == 'parameter':
                    self.parameters.append(c)
                    self.parameter_ranges[c] = eval(pr[2])
                elif pr[0] == 'objective':
                    self.objectives.append(c)
                    self.indicators[c] = pr[2]
            cols.append(c)
        df.columns = cols
        return df

    from_project = from_csv

    def to_csv(self, *args, **kwargs):
        """Put parameter ranges back into columns and save as csv.
        """
        df = self.copy()
        cols = []
        for c in df.columns:
            if c in self.parameters:
                c = 'parameter:'+c+':%r' % (self.parameter_ranges[c],)
            elif c in self.objectives:
                c = 'objective:'+c+':%s' % self.indicators[c]
            cols.append(c)
        df.columns = cols
        df.to_csv(*args, **kwargs)
        return

    def to_run(self, run, tags=''):
        """Save with run as compressed csv.
        """
        tags = (tags+' ' if tags else '')+'optimization_populations'
        fn = osp.basename(osp.splitext(self.path)[0])+'.csv.gzip'
        tmpf = osp.join(self.project.browser.settings.tmpfilesdir, fn)
        self.to_csv(tmpf, compression='gzip')
        f = self.project.browser.insert('file', run=run, tags=tags, file=tmpf)
        return f

    @property
    def last_generation(self):
        return self.loc[max(self.index.levels[0])]
    # alias
    lastgen = last_generation

    @plot_function
    def plot_generation_objectives(self, ax=None, output=None, **kw):
        """Show the median (min-max) objective values over all generations.
        """
        genperf = self[self.objectives].groupby(axis=0, level=0)
        axs = genperf.median().plot(subplots=True, ax=ax, legend=False, rot=0,
                                    sharex=True, title=self.objectives)
        for a, col in zip(axs, self.objectives):
            color = a.get_lines()[0].get_color()
            a.fill_between(genperf.groups.keys(), genperf.min()[col],
                           genperf.max()[col], color=color, alpha=0.15)
        plt.tight_layout()
        return axs

    @plot_function
    def plot_objective_scatter(self, generation=None, best=None,
                               selected=None, selected_color='r', ax=None,
                               runs=None, output=None, **scatterkwargs):
        """Plot all objectives against each other in a stepped subplot.

        Arguments
        ---------
        generation : int, optional
            The generation to plot objectives from. Default: last.
        best : bool | min. objectives
            Highlight the best tradeoff solution. Takes precendence over
            selected.
        selected : dict-like
            Highlight selected point(s).
        selected_color : matplotlib.color spec | str
            Color for the selected points.
        scatterkwargs :
            Any keyword passed onto the scatter function.
        """
        # get objectives to plot
        gen = self.loc[generation] if generation else self.lastgen

        if best is not None:
            selected = self.best_tradeoff(best)

        ax = plot_objective_scatter(gen[self.objectives], selected=selected,
                                    ax=ax, **scatterkwargs)

        return ax

    @plot_function
    def plot_parameter_distribution(self, parameters=None, generation=None,
                                    runs=None, ax=None, output=None, **histkw):
        '''Plot parameter distribution histograms of all parameters.

        Arguments
        ---------
        parameters : list
            Limit what parameters to show. Default: all
        generation : int, optional
            Generation to plot. Default: last
        histkw :
            Any keyword parsed to the plt.hist call (excpt for ``range``) and
            the subsequent Patch. A useful keyword for multiple calls is
            ``histtype='step'`` which is the default if runs are parsed.
        '''
        gen = self.loc[generation] if generation else self.lastgen
        parameters = parameters or self.parameters
        # setup nice plots
        sq = np.sqrt(len(parameters))
        ncols = int(sq)
        nrows = ncols + (1 if sq-ncols else 0)
        if ax:
            f = ax.get_figure()
            axs = f.get_axes()
            if len(axs) == ncols*nrows:
                ax = np.array(axs)
            else:
                f.clear()
                ax = None
        else:
            f = plt.figure()
        if ax is None:
            ax = f.subplots(nrows, ncols, squeeze=False, sharey=True).flatten()
        histkw.setdefault("bins", 10)
        if runs:
            histkw.setdefault('histtype', 'step')
        hdls = []
        for a, par in zip(ax, parameters):
            bars = a.hist(gen[par], range=self.parameter_ranges[par],
                          **histkw)
            hdls.append(bars)
            a.set_title(par)
            a.set_xlim(*self.parameter_ranges[par])
        ax[0].set_ylabel('N runs')
        # remove not needed axes
        for i in range(len(parameters), ncols*nrows):
            ax[i].set_axis_off()
        plt.tight_layout()
        return hdls

    def best_tradeoff(self, minobjectives=None):
        '''Select from last generation the parameter with the shortest distance
        to the scaled Pareto front to the origin. The front is either scaled
        by the max of each objective dimension or by minobjectives, which
        should be a dictionary of objectives to take account for.'''
        if minobjectives is not None:
            minobjectives = pd.Series(minobjectives)
        else:
            # scale by objective maximum
            minobjectives = self.lastgen[self.objectives].max()
        scobs = self.lastgen[self.objectives] / minobjectives
        # distance to origin
        dist = np.sqrt((scobs**2).sum(1))
        # store best parameter set with the lowest distance
        best = self.lastgen.loc[dist.idxmin()]
        return best

    def select_min_objectives(self, minobjectives=None, **minobjkwargs):
        if minobjectives is not None:
            msg = 'minobjectives must be the same length as objectives'
            assert len(minobjectives) == len(self.objectives), msg
            minobjkwargs.update(dict(zip(self.objectives, minobjectives)))

        lg = self.lastgen.copy()
        for o, v in minobjkwargs.items():
            lg = lg[lg[o] < v]

        return lg
