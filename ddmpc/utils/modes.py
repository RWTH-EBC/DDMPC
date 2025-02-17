#!/usr/bin/env python

""" modes.py: Modes for the Control features. """

import abc
import datetime
import random
from typing import Optional

import numpy
import numpy as np


class Mode(abc.ABC):

    def __init__(
            self,
            day_start: int = 8,  # default
            day_end: int = 16,  # default
    ):
        self.day_start: int = day_start
        self.day_end: int = day_end

    def __str__(self):
        return f'Mode({self.__class__.__name__})'

    def __repr__(self):
        return f'Mode({self.__class__.__name__})'

    @abc.abstractmethod
    def error(self, time: int, value: float) -> float:
        """ Returns the control error """
        ...

    @abc.abstractmethod
    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """
        ...

    def lb(self, time: int) -> float:
        return self.bounds(time)[0]

    def ub(self, time: int) -> float:
        return self.bounds(time)[1]

    @abc.abstractmethod
    def target(self, time: int) -> float:
        """ returns the control target for a given time """
        ...

    def _day(self, time: int) -> bool:
        """ returns True if the given time is during day (defined by day_start and day_end) """

        time = datetime.datetime.fromtimestamp(time)

        return self.day_start <= time.hour < self.day_end

    @staticmethod
    def _weekend(time: int) -> bool:
        """ returns True if the given time is during weekend """

        time = datetime.datetime.fromtimestamp(time)

        return 5 <= time.weekday()


class Steady(Mode):
    """
    steady set points for day and night
    day defined by day_start and day_end
    """

    def __init__(
            self,
            day_start: int = 8,
            day_end: int = 16,
            day_target: float = 273.15 + 20,
            night_target: float = 273.15 + 18,
            weekend: bool = True,
    ):
        """
        steady set point for day and night

        :param day_start: start time of day
        :param day_end: end time of day
        :param day_target: set point during the day (day_start until day_end)
        :param night_target: set point during the night
        :param weekend: set True if on weekend days the night boundaries should be used
        """

        super(Steady, self).__init__(day_start=day_start, day_end=day_end)

        self.day_target = day_target
        self.night_target = night_target
        self.weekend: bool = weekend

    def error(self, value: float, time: int) -> float:
        """ Returns the control error at a given time"""

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """
        returns the lower and upper bound at a given time
        since there are no upper and lower bounds in steady mode, it returns NaN
        """

        return numpy.nan, numpy.nan

    def target(self, time: int) -> float:
        """
        returns the control target for a given time
        returns day_target only during weekdays, otherwise night_target
        """

        if self._weekend(time) and self.weekend:
            return self.night_target

        if self._day(time):
            return self.day_target

        return self.night_target


class Random(Mode):
    """
    random sequence of set points between bounds
    day defined by day_start and day_end
    different bounds for day and night
    """

    def __init__(
            self,
            day_start: int = 8,
            day_end: int = 16,
            day_lb: float = 273.15 + 19,
            night_lb: float = 273.15 + 16,
            day_ub: float = 273.15 + 21,
            night_ub: float = 273.15 + 24,
            interval: int = 60 * 60 * 4,
            weekend: bool = True,
    ):
        """
        random sequence of set points between given bounds

        :param day_start: start time of day
        :param day_end: end time of day
        :param day_lb: lower bound for day (day_start until day_end)
        :param night_lb: lower bound for night
        :param day_ub: upper bound for day (day_start until day_end)
        :param night_ub: upper bound for night
        :param interval: time interval between two randomly generated targets / set points
        :param weekend: set True if on weekend days the night boundaries should be used
        """
        super(Random, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb: float = day_lb
        self.night_lb: float = night_lb
        self.day_ub: float = day_ub
        self.night_ub: float = night_ub

        self.interval: int = interval
        self.weekend: bool = weekend

        self.last_randomization: Optional[int] = None  # at initialization there hasn't been a randomization yet
        self.current_target: Optional[int] = None   # at initialization there is no current target yet

    def error(self, value: float, time: int) -> float:
        """ Returns the control error at a given time"""

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """
        returns the lower and upper bound at a given time as tuple
        returns day bounds only during weekdays, otherwise night bounds
        """

        if self._weekend(time) and self.weekend:
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """
        returns the control target at a given time
        and returns a new target every [interval] seconds
        """

        lb, ub = self.bounds(time)

        if self.last_randomization is None:
            self.last_randomization = time

        if self.current_target is None:
            self.current_target = random.uniform(lb, ub)

        def new_target():
            """
            returns a new target in K within bounds
            the difference between the current target and new target is 1 K minimum
            """
            target = random.uniform(lb, ub)

            # if the change of the target would be lower than 1 K, randomize again
            if abs(self.current_target - target) < 1:
                target = new_target()

            return target

        # if the time [interval] passed since last randomization or there is no target yet or the target is not within
        # bounds, generate a new target and set current time as time of last randomization
        if time - self.last_randomization >= self.interval or \
                self.current_target is None or \
                self.current_target < lb or \
                ub < self.current_target:
            self.current_target = new_target()
            self.last_randomization = time

        return self.current_target


class Identification(Mode):
    """
    random sequence of set points between bounds in random intervals between bounds
    day defined by day_start and day_end
    different bounds for day and night
    """

    def __init__(
            self,
            day_start: int = 8,
            day_end: int = 16,
            day_lb: float = 273.15 + 19,
            night_lb: float = 273.15 + 16,
            day_ub: float = 273.15 + 21,
            night_ub: float = 273.15 + 24,

            min_interval: int = 60 * 60 * 2,
            max_interval: int = 60 * 60 * 5,
            min_change: float = 1,
            max_change: float = 2,
            weekend: bool = True,
    ):
        """
        random sequence of set points between given bounds in random intervals between given bounds

        :param day_start: start time of day
        :param day_end: end time of day
        :param day_lb: lower bound for day (day_start until day_end)
        :param night_lb: lower bound for night
        :param day_ub: upper bound for day (day_start until day_end)
        :param night_ub: upper bound for night
        :param min_interval: minimum time interval between two randomly generated targets / set points
        :param max_interval: maximum time interval between two randomly generated targets / set points
        :min_change: minimum change (absolute) between two targets / set points
        :max_change: maximum change (absolute) between two targets / set points
        :param weekend: set True if on weekend days the night boundaries should be used
        """

        super(Identification, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb: float = day_lb
        self.night_lb: float = night_lb
        self.day_ub: float = day_ub
        self.night_ub: float = night_ub

        self.max_interval: int = max_interval
        self.min_interval: int = min_interval
        self.interval: int = random.randrange(min_interval, max_interval)
        self.min_change: float = min_change
        self.max_change: float = max_change
        self.weekend: bool = weekend

        self.last_randomization: Optional[int] = None       # at initialization there hasn't been a randomization yet
        self.current_target: Optional[int] = None           # at initialization there is no current target yet

    def error(self, value: float, time: int) -> float:
        """ Returns the control error """

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """
        returns the lower and upper bound at a given time as tuple
        returns day bounds only during weekdays, otherwise night bounds
        """

        if self._weekend(time) and self.weekend:
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """
        returns the control target at a given time
        and returns a new target every [interval] seconds
        interval differs randomly between min_interval and max_interval
        """

        lb, ub = self.bounds(time)

        if self.last_randomization is None:
            self.last_randomization = time

        if self.current_target is None:
            self.current_target = random.randint(int(lb), int(ub))

        def new_target():
            """
            returns a new target in K
            randomly choosing if positive or negative change is applied whit respect to bounds
            """

            # min / max positive change with respect to upper bounds and min / max change
            max_pos_change = min(max(ub - self.current_target, 0), self.max_change)
            min_pos_change = min(max(ub - self.current_target, 0), self.min_change)

            # min / max negative change with respect to lower bounds and min / max change
            max_neg_change = min(max(self.current_target - lb, 0), self.max_change)
            min_neg_change = min(max(self.current_target - lb, 0), self.min_change)

            # choose pos / neg change randomly from determined intervals
            pos_change = random.uniform(a=min_pos_change, b=max_pos_change)
            neg_change = - random.uniform(a=min_neg_change, b=max_neg_change)

            if self.current_target + self.min_change > ub:
                # if new target would be out of upper bound if min_change would be added to current target
                c = neg_change
            elif self.current_target - self.min_change < lb:
                # if new target would be out of lower bound if min_change would be subtracted to current target
                c = pos_change
            else:
                # else pick pos or neg change randomly
                c = random.choice([pos_change, neg_change])

            target = self.current_target + c

            return target

        # if the time [interval] passed since last randomization or there is no target yet or the target is not within
        # bounds, generate a new interval and target and set current time as time of last randomization
        if time - self.last_randomization >= self.interval or \
                self.current_target is None or \
                self.current_target < lb or \
                ub < self.current_target:
            # new interval
            self.interval = int(random.uniform(self.min_interval, self.max_interval))

            self.current_target = new_target()
            self.last_randomization = time

        return self.current_target


class Economic(Mode):
    """
    sets bounds for day and night, no specific set point only boundaries given
    day defined by day_start and day_end
    different bounds for day and night
    """

    def __init__(
            self,
            day_start: int = 8,
            day_end: int = 16,
            day_lb: float = 273.15 + 19,
            day_ub: float = 273.15 + 22,
            night_lb: float = 273.15 + 16,
            night_ub: float = 273.15 + 25,
            weekend: bool = True,
    ):
        """
        sets bounds for day and night, no specific set point only boundaries given

        :param day_start: start time of day
        :param day_end: end time of day
        :param day_lb: lower bound for day (day_start until day_end)
        :param night_lb: lower bound for night
        :param day_ub: upper bound for day (day_start until day_end)
        :param night_ub: upper bound for night
        :param weekend: set True if on weekend days the night boundaries should be used
        """

        super(Economic, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb: float = day_lb
        self.night_lb: float = night_lb
        self.day_ub: float = day_ub
        self.night_ub: float = night_ub

        self.weekend: bool = weekend

    def error(self, value: float, time: int) -> float:
        """
        Returns the control error at a given time. Since there is no actual set point but only boundaries,
        the control error is 0 within bounds
        """

        lb, ub = self.bounds(time)

        if value < lb:
            return lb - value

        if value > ub:
            return ub - value

        return 0

    def bounds(self, time: int) -> tuple[float, float]:
        """
        returns the lower and upper bound for a given time
        If boolean weekend is True, on weekend days the night boundaries are used
        """

        if self._weekend(time) and self.weekend:
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """
        returns the control target for a given time
        since there is no actual set point but only boundaries given, here NaN is returned
        """

        return numpy.nan


class Power(Mode):

    def __init__(
            self,
    ):

        super(Power, self).__init__(day_start=8, day_end=16)

    def error(self, value: float, time: int) -> float:
        """ Returns the control error """

        lb, ub = self.bounds(time)

        if value < lb:
            return lb - value

        if value > ub:
            return ub - value

        return 0

    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """

        return np.nan, np.nan

    def target(self, time: int) -> float:
        """ returns the control target for a given time """

        return 0


class NoMode(Mode):

    def __init__(self):
        super(NoMode, self).__init__(day_start=np.nan, day_end=np.nan)

    def error(self, time: int, value: float) -> float:
        """ Returns the control error """
        return 0

    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """

        return np.nan, np.nan

    def target(self, time: int) -> float:
        """ returns the control target for a given time """
        return np.nan
