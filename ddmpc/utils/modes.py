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
            day_start:      int = 8,
            day_end:        int = 16,
    ):

        self.day_start: int = day_start
        self.day_end:   int = day_end


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
        """ returns True if the given time is during day """

        time = datetime.datetime.fromtimestamp(time)

        return self.day_start <= time.hour < self.day_end

    def _weekend(self, time: int) -> bool:
        """ returns True if the given time is during weekend """

        time = datetime.datetime.fromtimestamp(time)

        return 5 <= time.weekday()


class Steady(Mode):
    """ steady values for day and night """

    def __init__(
            self,
            day_start:      int = 8,
            day_end:        int = 16,
            day_target:     float = 273.15 + 20,
            night_target:   float = 273.15 + 18,
    ):

        super(Steady, self).__init__(day_start=day_start, day_end=day_end)

        self.day_target = day_target
        self.night_target = night_target

    def error(self, value: float, time: int) -> float:
        """ Returns the control error """

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """

        return numpy.nan, numpy.nan

    def target(self, time: int) -> float:
        """ returns the control target for a given time """

        if self._weekend(time):
            return self.night_target

        if self._day(time):
            return self.day_target

        return self.night_target


class Random(Mode):
    """ random sequence of set points between bounds """

    def __init__(
            self,
            day_start:      int = 8,
            day_end:        int = 16,
            day_lb:         float = 273.15 + 19,
            night_lb:       float = 273.15 + 16,
            day_ub:         float = 273.15 + 21,
            night_ub:       float = 273.15 + 24,
            interval:       int = 60 * 60 * 4,
    ):

        super(Random, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb:    float = day_lb
        self.night_lb:  float = night_lb
        self.day_ub:    float = day_ub
        self.night_ub:  float = night_ub

        self.interval:              int = interval
        self.last_randomization:    Optional[int] = None
        self.current_target:        Optional[int] = None

    def error(self, value: float, time: int) -> float:
        """ Returns the control error """

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """

        if self._weekend(time):
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """ returns the control target for a given time """

        lb, ub = self.bounds(time)

        if self.last_randomization is None:
            self.last_randomization = time

        if self.current_target is None:
            self.current_target = random.uniform(lb, ub)

        def new_target():
            target = random.uniform(lb, ub)

            if abs(self.current_target - target) < 1:
                target = new_target()

            return target

        if time - self.last_randomization >= self.interval or\
                self.current_target is None or\
                self.current_target < lb or\
                ub < self.current_target:

            self.current_target = new_target()
            self.last_randomization = time

        return self.current_target


class Identification(Mode):
    """ random sequence of set points between bounds """

    def __init__(
            self,
            day_start:      int = 8,
            day_end:        int = 16,
            day_lb:         float = 273.15 + 19,
            night_lb:       float = 273.15 + 16,
            day_ub:         float = 273.15 + 21,
            night_ub:       float = 273.15 + 24,

            min_interval:   int = 60 * 60 * 2,
            max_interval:   int = 60 * 60 * 5,
            min_change:     int = 1,
            max_change:     int = 2,
    ):

        super(Identification, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb:    float = day_lb
        self.night_lb:  float = night_lb
        self.day_ub:    float = day_ub
        self.night_ub:  float = night_ub

        self.max_interval:          int = max_interval
        self.min_interval:          int = min_interval
        self.interval:              int = random.randrange(min_interval, max_interval)
        self.min_change:            int = min_change
        self.max_change:            int = max_change

        self.last_randomization:    Optional[int] = None
        self.current_target:        Optional[int] = None

    def error(self, value: float, time: int) -> float:
        """ Returns the control error """

        return self.target(time) - value

    def bounds(self, time: int) -> tuple[float, float]:
        """ returns the lower and upper bound for a given time """

        if self._weekend(time):
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """ returns the control target for a given time """

        lb, ub = self.bounds(time)

        if self.last_randomization is None:
            self.last_randomization = time

        if self.current_target is None:
            self.current_target = random.randint(int(lb), int(ub))

        def new_target():

            max_pos_change = min(abs(ub - self.current_target), self.max_change)
            min_pos_change = min(abs(ub - self.current_target), self.min_change)

            max_neg_change = min(abs(lb - self.current_target), self.max_change)
            min_neg_change = min(abs(lb - self.current_target), self.min_change)

            pos_change = random.uniform(a=min_pos_change, b=max_pos_change)
            neg_change = - random.uniform(a=min_neg_change, b=max_neg_change)

            if self.current_target + self.min_change > ub:
                c = neg_change
            elif self.current_target - self.min_change < lb:
                c = pos_change
            else:
                c = random.choice([pos_change, neg_change])

            target = self.current_target + c

            return target

        if time - self.last_randomization >= self.interval or\
                self.current_target is None or\
                self.current_target < lb or\
                ub < self.current_target:

            # new interval
            self.interval = int(random.uniform(self.min_interval, self.max_interval))

            self.current_target = new_target()
            self.last_randomization = time

        return self.current_target


class Economic(Mode):
    """ bounds for day and night """

    def __init__(
            self,
            day_start:  int = 8,
            day_end:    int = 16,
            day_lb:     float = 273.15 + 19,
            day_ub:     float = 273.15 + 22,
            night_lb:   float = 273.15 + 16,
            night_ub:   float = 273.15 + 25,
            weekend:    bool = True,
    ):

        super(Economic, self).__init__(day_start=day_start, day_end=day_end)

        self.day_lb:    float = day_lb
        self.night_lb:  float = night_lb
        self.day_ub:    float = day_ub
        self.night_ub:  float = night_ub

        self.weekend:   bool = weekend

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

        if self._weekend(time) and self.weekend:
            return self.night_lb, self.night_ub

        if self._day(time):
            return self.day_lb, self.day_ub

        return self.night_lb, self.night_ub

    def target(self, time: int) -> float:
        """ returns the control target for a given time """

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
