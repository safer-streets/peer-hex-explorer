from calendar import monthrange
from typing import Literal, Self

CrimeType = Literal[
    "Drugs",
    "Shoplifting",
    "Robbery",
    "Burglary",
    "Criminal damage and arson",
    "Anti-social behaviour",
    "Bicycle theft",
    "Violence and sexual offences",
    "Theft from the person",
    "Other crime",
    "Public order",
    "Vehicle crime",
    "Other theft",
    "Possession of weapons",
]

Force = Literal[
    "Avon and Somerset",
    "Bedfordshire",
    "Cambridgeshire",
    "Cheshire",
    "City of London",
    "Cleveland",
    "Cumbria",
    "Derbyshire",
    "Devon and Cornwall",
    "Dorset",
    "Durham",
    "Dyfed Powys",
    "Essex",
    "Gloucestershire",
    "Greater Manchester",
    "Gwent",
    "Hampshire",
    "Hertfordshire",
    "Humberside",
    "Kent",
    "Lancashire",
    "Leicestershire",
    "Lincolnshire",
    "Merseyside",
    "Metropolitan",
    "Norfolk",
    "North Wales",
    "North Yorkshire",
    "Northamptonshire",
    # "Northern Ireland",  # whilst crime data is available, spatial and other data are not consistent with other forces
    "Northumbria",
    "Nottinghamshire",
    "South Wales",
    "South Yorkshire",
    "Staffordshire",
    "Suffolk",
    "Surrey",
    "Sussex",
    "Thames Valley",
    "Warwickshire",
    "West Mercia",
    "West Midlands",
    "West Yorkshire",
    "Wiltshire",
]

DEFAULT_FORCE: Force = "West Yorkshire"


def fix_force_name(force: Force) -> str:
    """Only use this for the PFA boundary shapefile"""
    NAME_ADJUSTMENTS = {
        "Metropolitan": "Metropolitan Police",
        "Devon and Cornwall": "Devon & Cornwall",
        "City of London": "London, City of",
        "Dyfed Powys": "Dyfed-Powys",
    }
    return NAME_ADJUSTMENTS.get(force, force)


class Month:
    def __init__(self, y: int, m: int) -> None:
        assert 1900 <= y <= 2100
        assert 1 <= m <= 12
        self.y = y
        self.m = m - 1  # internally use 0-11

    @property
    def year(self) -> int:
        return self.y

    @property
    def month(self) -> int:
        return self.m + 1

    @property
    def days(self) -> int:
        "Returns no of days in month"
        return monthrange(self.y, self.m + 1)[1]

    @staticmethod
    def parse_str(yyyy_mm: str) -> Month:
        return Month(*(int(n) for n in yyyy_mm.split("-")))

    def __add__(self, months: int) -> Month:
        if months < 0:
            return self - abs(months)
        m = self.m + months
        years = m // 12
        m = m % 12
        y = self.y + years
        return Month(y, m + 1)

    def __sub__(self, months: int) -> Month:
        if months < 0:
            return self + abs(months)
        m = self.m - months
        years = m // 12  # this will be negative if m < 0
        m = m % 12
        y = self.y + years
        return Month(y, m + 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.y == other.y and self.m == other.m

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.y < other.y or (self.y == other.y and self.m < other.m)

    def __ge__(self, other: Self) -> bool:
        return not self < other

    def __repr__(self) -> str:
        return f"{self.y}-{self.m + 1:02}"

    def __hash__(self) -> int:
        return hash(self.y * 12 + self.m)
