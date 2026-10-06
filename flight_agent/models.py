"""Immutable task policies; these are data, never instructions from tool output."""
from dataclasses import asdict, dataclass
from datetime import date


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class Constraints:
    origin: str = 'SGN'
    destination: str = 'HAN'
    departure_date: str = '2026-11-20'
    passengers: int = 1
    max_total_price: int = 2_000_000
    max_stops: int = 1
    min_baggage_kg: int = 0
    allowed_airlines: tuple[str, ...] = ()

    def __post_init__(self):
        for airport in (self.origin, self.destination):
            if not isinstance(airport, str) or len(airport) != 3 or not airport.isupper() or not airport.isalpha():
                raise ValueError('Airports must be three uppercase letters')
        if self.origin == self.destination:
            raise ValueError('Origin and destination must differ')
        try:
            date.fromisoformat(self.departure_date)
        except (TypeError, ValueError) as exc:
            raise ValueError('departure_date must be ISO YYYY-MM-DD') from exc
        positive_int(self.passengers, 'passengers')
        positive_int(self.max_total_price, 'max_total_price')
        positive_int(self.max_stops, 'max_stops', 0)
        positive_int(self.min_baggage_kg, 'min_baggage_kg', 0)
        if not isinstance(self.allowed_airlines, (tuple, list)) or any(
                not isinstance(a, str) or not a for a in self.allowed_airlines):
            raise ValueError('allowed_airlines must contain airline names')
        object.__setattr__(self, 'allowed_airlines', tuple(self.allowed_airlines))


@dataclass(frozen=True)
class Task:
    task_id: str
    constraints: Constraints
    objective: str = 'recommend'

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError('task_id is required')
        if self.objective not in ('recommend', 'book'):
            raise ValueError('objective must be recommend or book')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(data['task_id'], Constraints(**data['constraints']), data['objective'])


@dataclass(frozen=True)
class Approval:
    task_id: str
    principal: str
    flight_id: str
    passengers: int
    max_total_price: int

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in
               (self.task_id, self.principal, self.flight_id)):
            raise ValueError('Approval identifiers are required')
        positive_int(self.passengers, 'passengers')
        positive_int(self.max_total_price, 'max_total_price')


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def ok(data):
    return {'ok': True, 'data': data}


def error(code, message):
    return {'ok': False, 'error': code, 'message': message}
