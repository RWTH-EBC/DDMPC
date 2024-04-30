import time
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Iterator
from typing import Union

import numpy as np
import pandas as pd
from casadi import MX, inf, nlpsol, vertcat

import ddmpc.utils.formatting as fmt
from ddmpc.controller.model_predictive.costs import Cost, AbsoluteLinear
from ddmpc.modeling.features.features import Feature, Source, Constructed, Controlled, Control
from ddmpc.modeling.modeling import Model
from ddmpc.modeling.predicting import Predictor
from ddmpc.utils.modes import Economic, Steady


class Objective:
    """ objective function for the optimization problem """

    def __init__(
            self,
            feature:        Feature,
            cost:           Cost,
            ignore_mode:    bool = False,
    ):
        """
        :param feature: feature to weight in the cost function
        :param cost: cost function to apply to the feature
        :param k_0_offset: offset for the first value(s) of the Feature (e.g. first state is measured and cannot be
                           changed -> therefore no consideration in cost function)
        :param k_N_offset: offset for the last value(s) of the feature (e.g. last control has no effect)
        """

        self.feature:       Feature = feature
        self.cost:          Cost = cost
        self.ignore_mode:   bool = ignore_mode

    def __str__(self):
        return f'Objective(feature={self.feature})'

    def __call__(self, mx: MX) -> MX:

        return self.cost(mx)


class Constraint:
    """ constraint for the optimization problem """

    def __init__(
            self,
            feature:     Feature,
            lb:         float,
            ub:         float,
    ):
        """
        :param feature: feature to add constraints to
        :param lb: lower bound for the constraint
        :param ub: upper bound for the constraint
        :param k_0_offset: offset for the first value(s) of the Feature (e.g. first state is measured and cannot be
            changed -> therefore no consideration in cost function)
        :param k_N_offset: offset for the last value(s) of the feature (e.g. last control has no effect)
        """

        self.feature:       Feature = feature

        self.lb:            float = lb
        self.ub:            float = ub

    def get(self, k: int) -> 'NLPConstraint':

        return NLPConstraint(
            expression=self.feature.source[k],
            lb=self.lb,
            ub=self.ub,
        )

    def __str__(self):
        return f'Constraint(features={self.feature}, lb={self.lb}, ub={self.ub})'

    def __repr__(self):
        return f'Constraint(features={self.feature}, lb={self.lb}, ub={self.ub})'


class NLPVariable(ABC):

    def __init__(
            self,
            feature:    Feature,
            k:          int,
    ):
        """ variable for the nlp """

        self.feature = feature
        self.k = k

        # call the mx variable to instantiate it
        a = self.feature.source[self.k]

    @property
    @abstractmethod
    def mx(self) -> MX:
        pass

    @property
    @abstractmethod
    def col_name(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):

        if isinstance(other, NLPVariable):

            return self.feature == other.feature and type(self) == type(other) and self.k == other.k

        return False

    def __ne__(self, other):
        return not self == other


class NLPValue(NLPVariable):

    def __init__(
            self,
            feature:    Feature,
            k:          int,
    ):

        super(NLPValue, self).__init__(feature=feature, k=k)

    @property
    def mx(self) -> MX:
        return self.feature.source[self.k]

    @property
    def col_name(self):
        return self.feature.source.col_name

    def __str__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    def __repr__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'


class NLPTarget(NLPVariable):

    def __init__(
            self,
            feature:    Controlled,
            k:          int,
    ):
        super(NLPTarget, self).__init__(feature=feature, k=k)

        self._mx: MX = MX.sym(f'MX({self.__class__.__name__}({self.feature}) at k={"%+d" % k})')

    @property
    def mx(self) -> MX:
        return self._mx

    @property
    def col_name(self):

        self.feature: Controlled
        return self.feature.col_name_target

    def __str__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    def __repr__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'


class NLPLowerBound(NLPVariable):

    def __init__(
            self,
            feature:    Controlled,
            k:          int,
    ):
        super(NLPLowerBound, self).__init__(feature=feature, k=k)

        self._mx: MX = MX.sym(f'{self.__class__.__name__}({self.feature})[{"%+d" % k}]')

    @property
    def mx(self) -> MX:
        return self._mx

    @property
    def col_name(self):
        self.feature: Controlled
        return self.feature.col_name_lb

    def __str__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    def __repr__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'


class NLPUpperBound(NLPVariable):

    def __init__(
            self,
            feature: Controlled,
            k: int,
    ):
        super(NLPUpperBound, self).__init__(feature=feature, k=k)

        self._mx: MX = MX.sym(f'{self.__class__.__name__}({self.feature})[{"%+d" % k}]')

    def __str__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    def __repr__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    @property
    def mx(self) -> MX:
        return self._mx

    @property
    def col_name(self):
        self.feature: Controlled
        return self.feature.col_name_ub


class NLPEpsilon(NLPVariable):

    def __init__(
            self,
            feature: Feature,
            k: int,
    ):
        super(NLPEpsilon, self).__init__(feature=feature, k=k)

        self._mx: MX = MX.sym(f'{self.__class__.__name__}({self.feature})[{"%+d" % k}]')

    def __str__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    def __repr__(self):
        return f'{__class__.__name__}({self.feature}[{"%+d" % self.k}])'

    @property
    def mx(self) -> MX:
        return self._mx

    @property
    def col_name(self):
        return f'Epsilon(feature={self.feature}, k={self.k})'


class NLPConstraint:

    def __init__(
            self,
            expression: MX,
            lb: float = 0.0,
            ub: float = 0.0,
    ):
        """ constraint for the nlp """

        self.expression:    MX = expression
        self.lb:            float = lb
        self.ub:            float = ub

    def __str__(self):
        return f'{self.__class__.__name__}({self.lb} < {self.expression} < {self.ub})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.lb} < {self.expression} < {self.ub})'


class NLPObjective:

    def __init__(
            self,
            expression: Union[MX, float],
    ):
        self.expression = expression

    def __str__(self):
        return f'{self.__class__.__name__}({self.expression})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expression})'


class NLPSolution:

    def __init__(
            self,
            par_vars:   list,
            opt_vars:   list,
            par_vals:   list,
            opt_vals:   list,
            inp_map:    dict[Predictor, dict[int, list[NLPVariable]]],

            success:    bool,
            status:     str,
            runtime:    float,
    ):
        assert len(par_vals) == len(par_vars)
        assert len(opt_vals) == len(opt_vars)

        self.par_vals = par_vals
        self.opt_vals = opt_vals

        self.par: Iterator = zip(par_vars, par_vals)
        self.opt: Iterator = zip(opt_vars, opt_vals)

        self.hashmap = dict()
        for var, val in zip(par_vars + opt_vars, par_vals + opt_vals):

            if isinstance(var, NLPEpsilon):
                continue

            if var.col_name not in self.hashmap:
                self.hashmap[var.col_name] = dict()

            self.hashmap[var.col_name][var.k] = float(val)

        self.inp_vals = inp_map

        self.success: bool = success
        self.status:  str = status
        self.runtime: float = runtime

    def __str__(self):
        return f'NLPSolution(runtime={self.runtime})'

    def __repr__(self):
        return f'NLPSolution(runtime={self.runtime})'

    @property
    def optimal_controls(self) -> dict[str, float]:
        controls = dict()

        for var, val in self.opt:
            condition1 = isinstance(var.feature, Control)
            condition2 = var.k == 0
            condition3 = isinstance(var, NLPValue)

            if condition1 and condition2 and condition3:
                if var.feature.cutoff is not None:
                    if val <= var.feature.cutoff:
                        val = var.feature.default
                controls[var.col_name] = val

        return controls

    def value(self, nlp_val: NLPVariable) -> float:

        return self.hashmap[nlp_val.col_name][nlp_val.k]

    @property
    def df(self) -> pd.DataFrame:
        """ turns the solution back to a DataFrame """

        df = pd.DataFrame(self.hashmap)
        df.sort_index(inplace=True)

        return df

    """
    def predictions(self, predictions: list[Prediction]):

        return {p.col_name: self.hashmap[p.feature.col_name][p.k] for p in predictions}
    """

    def summary(self):

        print()
        print('------------------------ NLPSolution ------------------------')

        rows = list()

        rows.append(['Par Vars'])
        rows.append(['    ', 'Type', 'Feature', 'k', 'value'])
        for var, val in self.par:
            rows.append(['        ', var.__class__.__name__, var.feature, var.k, round(val, 4)])

        rows.append(['Opt Vars'])
        rows.append(['    ', 'Type', 'Feature', 'k', 'value'])
        for var, val in self.opt:
            rows.append(['        ', var.__class__.__name__, var.feature, var.k, round(val, 4)])
        fmt.print_table(rows)

        print()


class NLP:


    def __init__(
            self,
            N:              int,
            model:          Model,
            control_change_step: int = 1,
            objectives:     list[Objective] = None,
            constraints:    list[Constraint] = None,
    ):

        self.model:             Model = model
        self.objectives:        list[Objective] = objectives
        self.constraints:       list[Constraint] = constraints

        self._par_vars:         list[NLPVariable] = list()
        self._opt_vars:         list[NLPVariable] = list()
        self._var_map:          dict[tuple[Source, int], NLPVariable] = dict()
        self._inp_map:          dict[Predictor, dict[int, list[NLPVariable]]] = dict()

        self._constraints:      list[NLPConstraint] = list()
        self._objectives:       list[NLPObjective] = list()

        self.solution:          Optional[NLPSolution] = None

        self.model:             Model = model
        self.max_lag:           Optional[int] = None
        self.N:                 int = N
        assert N % control_change_step == 0, "NLP Horizon N must be a multiple of control change step!"
        self.control_change_step = control_change_step

        self.lastSolutionFailed = True

    def _map_indices(self, *predictors: Predictor):
        """ Creates a dict with a key for every Source and their respective start index """

        self.idx_map: dict[Union[Source, Feature], int] = dict()

        def assign(source: Source):

            if isinstance(source, Constructed):

                for sub in source.subs:
                    if sub in self.idx_map:
                        self.idx_map[sub] = max(self.idx_map[sub], self.idx_map[source] + source.past_steps)
                    else:
                        self.idx_map[sub] = self.idx_map[source] + source.past_steps

                    assign(sub)

        for feature in self.model.features:

            self.idx_map[feature.source] = 0

            assign(feature.source)

        for predictor in predictors:

            for inp in predictor.inputs:

                if inp.source in self.idx_map:
                    self.idx_map[inp.source] = max(self.idx_map[inp.source], inp.lag - 1)
                else:
                    self.idx_map[inp.source] = inp.lag - 1

                assign(inp.source)

    def is_variable(self, source: Union[Feature, Source], k: int) -> bool:

        return (source, k) in self._var_map.keys()

    def _add_variables(self):

        for x in self.model.controlled:

            for k in range(-self.idx_map[x.source], 1):
                self._add_par_var(NLPValue(feature=x, k=k))

            for k in range(1, self.N + 1):
                self._add_opt_var(NLPValue(feature=x, k=k))

        for u in self.model.controls:

            for k in range(-self.idx_map[u.source], 0):
                self._add_par_var(NLPValue(feature=u, k=k))

            for k in range(0, self.N + 1):
                if k % self.control_change_step == 0:
                    self._add_opt_var(NLPValue(feature=u, k=k))
                else:
                    self._var_map[u.source, k] = self._var_map[u.source, k - (k % self.control_change_step)]

        for d in self.model.disturbances:

            for k in range(-self.idx_map[d.source], self.N + 1):
                self._add_par_var(NLPValue(feature=d, k=k))

        for c in self.model.connecting:

            assert isinstance(c.source, Constructed)

            for k in range(-self.idx_map[c.source], 0):

                self._add_par_var(NLPValue(feature=c, k=k))

            for k in range(0, self.N + 1):

                self._add_opt_var(NLPValue(feature=c, k=k))

    def _add_par_var(self, par_var: NLPVariable):

        self._par_vars.append(par_var)
        self._var_map[par_var.feature.source, par_var.k] = par_var

    def _add_opt_var(self, opt_var: NLPVariable):

        self._opt_vars.append(opt_var)
        self._var_map[opt_var.feature.source, opt_var.k] = opt_var

    def _connect_constructed(self):

        for c in self.model.connecting:

            assert isinstance(c.source, Constructed)

            for k in range(0, self.N + 1):

                if isinstance(c.source, Controlled):
                    continue

                self._constraints.append(
                    NLPConstraint(expression=c.source.constraint(k))
                )

    def _add_predictions(self, *predictors: Predictor):

        for predictor in predictors:

            self._inp_map[predictor] = dict()

            for k in range(1, self.N + 1):

                # get the inputs for the predictor
                input_list = list()
                for inp in predictor.inputs:

                    # iterate backwards over the lag
                    for i in range(k - 1, k - inp.lag - 1, -1):

                        assert (inp.source, i) in self._var_map.keys(),\
                            KeyError(f'Please make sure {(inp.source, i)} is part of the Model. max_lag={self.idx_map[inp.source]}')

                        nlp_var = self._var_map[inp.source, i]

                        input_list.append(nlp_var)

                # input vars for Extrapolation Detector
                self._inp_map[predictor][k] = input_list

                prediction = predictor.predict([inp.mx for inp in input_list])[0]

                # access the output mx through the get_var() method to get the correct mx
                output_mx = self._var_map[predictor.output.source, k].mx

                # constraint
                self._constraints.append(
                    NLPConstraint(
                        expression=output_mx - prediction,
                        lb=0,
                        ub=0,
                    )
                )

    def _add_constraints(self, *constraints: Constraint):

        for constraint in constraints:

            for k in range(-self.max_lag, self.N + 1):

                # check if the MX exists in the nlp otherwise continue
                if not self.is_variable(constraint.feature.source, k):
                    continue

                self._constraints.append(
                    constraint.get(k)
                )

    def _add_objectives(self, *objectives: Objective):

        for objective in objectives:

            assert objective.feature.source in self.model.features, \
                f'The source of the {objective} is not included in the Model'

            for k in range(0, self.N + 1):

                # check if the variable exists for the given k otherwise continue
                if not self.is_variable(objective.feature.source, k):
                    continue

                nlp_value = self._var_map[objective.feature.source, k]
                feature = objective.feature

                if isinstance(feature, Controlled):

                    if isinstance(feature.mode, Economic):

                        lb_eps = NLPEpsilon(feature=feature, k=k)
                        ub_eps = NLPEpsilon(feature=feature, k=k)
                        self._opt_vars.extend((lb_eps, ub_eps))

                        lb = NLPLowerBound(feature=feature, k=k)
                        ub = NLPUpperBound(feature=feature, k=k)
                        self._par_vars.extend((lb, ub))

                        self._constraints.append(
                            NLPConstraint(nlp_value.mx - lb.mx + lb_eps.mx, 0, inf)
                        )
                        self._constraints.append(
                            NLPConstraint(nlp_value.mx - ub.mx + ub_eps.mx, -inf, 0)
                        )

                        # objective
                        self._objectives.append(
                            NLPObjective(objective(lb_eps.mx))
                        )
                        self._objectives.append(
                            NLPObjective(objective(ub_eps.mx))
                        )


                    elif isinstance(feature.mode, Steady):

                        target = NLPTarget(feature=feature, k=k)
                        self._par_vars.append(target)

                        # objective
                        self._objectives.append(
                            NLPObjective(objective(nlp_value.mx - target.mx))
                        )

                    else:
                        raise NotImplementedError(f'Mode {feature.mode} is not implemented yet '
                                                  f'for Objective {objective}.')

                elif isinstance(objective.cost, AbsoluteLinear):

                    eps1 = NLPEpsilon(feature=feature, k=k)
                    eps2 = NLPEpsilon(feature=feature, k=k)
                    self._opt_vars.append(eps1)
                    self._opt_vars.append(eps2)

                    # t1 - t2 = x
                    self._constraints.append(
                        NLPConstraint(expression=eps1.mx - eps2.mx - nlp_value.mx, lb=0, ub=0)
                    )
                    # eps1 > 0
                    self._constraints.append(
                        NLPConstraint(expression=eps1.mx, lb=0, ub=inf)
                    )
                    # eps2 > 0
                    self._constraints.append(
                        NLPConstraint(expression=eps2.mx, lb=0, ub=inf)
                    )

                    self._objectives.append(
                        NLPObjective(objective(eps1.mx))
                    )

                    self._objectives.append(
                        NLPObjective(objective(eps2.mx))
                    )

                else:

                    self._objectives.append(NLPObjective(objective(nlp_value.mx)))

    def _get_coldstart(self):

        cold_start_values: list = list()
        for opt_var in self._opt_vars:

            if isinstance(opt_var.feature, Control):
                cold_start_values.append(opt_var.feature.default)

            elif isinstance(opt_var.feature, Controlled):

                lb, ub = opt_var.feature.mode.bounds(0)
                target = opt_var.feature.mode.target(0)

                if target is not np.NAN:
                    cold_start_values.append(target)
                elif lb is not np.NAN and ub is not np.NAN:
                    cold_start_values.append((lb + ub) / 2)
                else:
                    cold_start_values.append(0)
                    warnings.warn(f'Mode for feature {opt_var.feature.source.name} does not provide lb, ub or target for cold start initialization, using 0 as default.')
            else:
                cold_start_values.append(0)
                warnings.warn(f'Cold start initialization for class {opt_var.feature.source} not implemented yet, using 0 as default.')
        return np.array(cold_start_values)

    def summary(self):

        print('------------------------- NLP SUMMARY -------------------------')

        print('Par Vars:')
        for par_var in self._par_vars:
            print('   ', par_var)
        print()

        print('Opt Vars:')
        for opt_var in self._opt_vars:
            print('   ', opt_var)
        print()

        print('Constraints:')
        for constraint in self._constraints:
            print('   ', constraint)
        print()

        print('Objectives:')
        for objective in self._objectives:
            print('   ', objective)
        print()

    def build(self, predictors: list[Predictor], alg: str = 'ipopt', solver_options: Optional[dict] = None):

        self.max_lag:           int = max([0] + [p.inputs.maxLag for p in predictors])

        self._par_vars:         list[NLPVariable] = list()
        self._opt_vars:         list[NLPVariable] = list()
        self._var_map:          dict[tuple[Source, int], NLPVariable] = dict()
        self._inp_map:          dict[Predictor, dict[int, list[NLPVariable]]] = dict()

        self._constraints:      list[NLPConstraint] = list()
        self._objectives:       list[NLPObjective] = list()

        self._map_indices(*predictors)

        self._add_variables()
        self._connect_constructed()
        self._add_predictions(*predictors)
        self._add_constraints(*self.constraints)
        self._add_objectives(*self.objectives)

        if solver_options is None:
            solver_options = dict()

        obj = sum([objective.expression for objective in self._objectives])

        g = [constraint.expression for constraint in self._constraints]

        par_vars = [par_var.mx for par_var in self._par_vars]
        opt_vars = [opt_var.mx for opt_var in self._opt_vars]

        nlp = {
            'x': vertcat(*opt_vars),
            'f': obj,
            'g': vertcat(*g),
            'p': vertcat(*par_vars),
        }

        self.solver = nlpsol('solver', alg, nlp, solver_options)

    def solve(self, par_vals: list[float]) -> NLPSolution:
        """ solves the nlp and stops the calculation time """

        assert self.solver is not None, 'Please make sure to call NLP.build() first.'

        lbg = [constraint.lb for constraint in self._constraints]
        ubg = [constraint.ub for constraint in self._constraints]

        nlp_instance = {
            'lbg': vertcat(*lbg),
            'ubg': vertcat(*ubg),
            'p': vertcat(*par_vals),
        }

        # warm start if a solution is available
        if self.solution is not None and self.lastSolutionFailed is False:
            nlp_instance['x0'] = self.solution.opt_vals
        else:
            nlp_instance['x0'] = self._get_coldstart()


        # call to the solver
        print('start solving')
        start_time = time.perf_counter()
        result = self.solver(**nlp_instance)
        stop_time = time.perf_counter()
        print('return_status:   ', self.solver.stats()['return_status'])
        print('success:         ', self.solver.stats()['success'])
        print('finished solving')

        if self.solver.stats()['return_status'] == 'Invalid_Number_Detected':
            self.lastSolutionFailed = True
        else:
            self.lastSolutionFailed = False


        self.solution = NLPSolution(
            par_vars=self._par_vars,
            opt_vars=self._opt_vars,
            par_vals=par_vals,
            opt_vals=[float(val) for val in result['x'].toarray()],
            inp_map=self._inp_map,
            runtime=stop_time - start_time,
            success=self.solver.stats()['success'],
            status=self.solver.stats()['return_status'],
        )

        return self.solution


