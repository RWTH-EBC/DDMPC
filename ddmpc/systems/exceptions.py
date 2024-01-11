
class ValidationError(Exception):

    def __init__(self, message):
        super().__init__(f'ValidationError: {message}')


class SimulationError(Exception):

    def __init__(self, message):
        super().__init__(f'SimulationError: {message}')


class ForecastError(Exception):

    def __init__(self, message):
        super().__init__(f'ForecastError: {message}')


class WritingError(Exception):

    def __init__(self, message):
        super().__init__(f'WritingError: {message}')


class ReadingError(Exception):

    def __init__(self, message):
        super().__init__(f'ReadingError: {message}')
