# dt_backend/core/market_hours.py
"""Market hours and holiday detection for US equity markets.

Checks:
- Weekends (Saturday, Sunday)
- US market holidays (New Year's, MLK, Presidents, Good Friday, Memorial, Independence, Labor, Thanksgiving, Christmas)
- Market hours (9:30 AM - 4:00 PM ET)
"""

from __future__ import annotations

from datetime import datetime, date, time
from typing import Optional
import pytz

# U.S. Equity markets timezone
NY_TZ = pytz.timezone("America/New_York")

# Market hours (regular session)
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)


def _get_us_market_holidays(year: int) -> set[date]:
    """Get US market holidays for a given year.
    
    Includes:
    - New Year's Day (observed)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents Day (3rd Monday in February)
    - Good Friday (Friday before Easter)
    - Memorial Day (last Monday in May)
    - Independence Day (observed)
    - Labor Day (1st Monday in September)
    - Thanksgiving Day (4th Thursday in November)
    - Christmas Day (observed)
    """
    holidays = set()
    
    # New Year's Day (observed)
    # If Jan 1 is Saturday, market is closed Friday Dec 31 (previous year)
    # If Jan 1 is Sunday, observed on Monday Jan 2
    new_year = date(year, 1, 1)
    if new_year.weekday() == 5:  # Saturday
        # Market closed on Friday Dec 31 of previous year
        # For this year's holidays, we just note Jan 1 is closed
        holidays.add(new_year)
    elif new_year.weekday() == 6:  # Sunday
        holidays.add(date(year, 1, 2))  # Observed Monday after
    else:
        holidays.add(new_year)
    
    # MLK Day - 3rd Monday in January
    jan_first = date(year, 1, 1)
    days_until_monday = (7 - jan_first.weekday()) % 7
    first_monday = jan_first.day + days_until_monday
    mlk_day = date(year, 1, first_monday + 14)  # 3rd Monday
    holidays.add(mlk_day)
    
    # Presidents Day - 3rd Monday in February
    feb_first = date(year, 2, 1)
    days_until_monday = (7 - feb_first.weekday()) % 7
    first_monday = feb_first.day + days_until_monday
    presidents_day = date(year, 2, first_monday + 14)  # 3rd Monday
    holidays.add(presidents_day)
    
    # Good Friday - Friday before Easter
    # Easter calculation using Meeus/Jones/Butcher algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(year, month, day)
    good_friday = date.fromordinal(easter.toordinal() - 2)
    holidays.add(good_friday)
    
    # Memorial Day - Last Monday in May
    may_31 = date(year, 5, 31)
    days_since_monday = (may_31.weekday() - 0) % 7
    memorial_day = date.fromordinal(may_31.toordinal() - days_since_monday)
    holidays.add(memorial_day)
    
    # Independence Day (observed)
    independence = date(year, 7, 4)
    if independence.weekday() == 5:  # Saturday
        holidays.add(date(year, 7, 3))  # Observed Friday before
    elif independence.weekday() == 6:  # Sunday
        holidays.add(date(year, 7, 5))  # Observed Monday after
    else:
        holidays.add(independence)
    
    # Labor Day - 1st Monday in September
    sep_first = date(year, 9, 1)
    days_until_monday = (7 - sep_first.weekday()) % 7
    labor_day = date(year, 9, sep_first.day + days_until_monday)
    holidays.add(labor_day)
    
    # Thanksgiving - 4th Thursday in November
    nov_first = date(year, 11, 1)
    days_until_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first.day + days_until_thursday
    thanksgiving = date(year, 11, first_thursday + 21)  # 4th Thursday
    holidays.add(thanksgiving)
    
    # Christmas Day (observed)
    christmas = date(year, 12, 25)
    if christmas.weekday() == 5:  # Saturday
        holidays.add(date(year, 12, 24))  # Observed Friday before
    elif christmas.weekday() == 6:  # Sunday
        holidays.add(date(year, 12, 26))  # Observed Monday after
    else:
        holidays.add(christmas)
    
    return holidays


def is_market_open(session_date: Optional[str] = None) -> bool:
    """Check if US equity market is open for the given date.
    
    Args:
        session_date: ISO date string (YYYY-MM-DD). If None, uses current NY time.
    
    Returns:
        True if market is open (weekday, not a holiday, during market hours)
        False if market is closed (weekend, holiday, or after-hours)
    """
    # Get current time in NY timezone
    now_ny = datetime.now(NY_TZ)
    
    # Parse session_date or use current date
    if session_date:
        try:
            check_date = datetime.fromisoformat(session_date).date()
        except (ValueError, AttributeError):
            # If parsing fails, use current date
            check_date = now_ny.date()
    else:
        check_date = now_ny.date()
    
    # Check if weekend (Saturday=5, Sunday=6)
    if check_date.weekday() >= 5:
        return False
    
    # Check if US market holiday
    holidays = _get_us_market_holidays(check_date.year)
    if check_date in holidays:
        return False
    
    # If checking for today, also check market hours
    if check_date == now_ny.date():
        current_time = now_ny.time()
        if current_time < MARKET_OPEN_TIME or current_time > MARKET_CLOSE_TIME:
            return False
    
    return True


def get_market_status(session_date: Optional[str] = None) -> dict[str, any]:
    """Get detailed market status for debugging.
    
    Args:
        session_date: ISO date string (YYYY-MM-DD). If None, uses current NY time.
    
    Returns:
        Dictionary with market status details
    """
    now_ny = datetime.now(NY_TZ)
    
    if session_date:
        try:
            check_date = datetime.fromisoformat(session_date).date()
        except (ValueError, AttributeError):
            check_date = now_ny.date()
    else:
        check_date = now_ny.date()
    
    is_weekend = check_date.weekday() >= 5
    holidays = _get_us_market_holidays(check_date.year)
    is_holiday = check_date in holidays
    
    is_today = check_date == now_ny.date()
    current_time = now_ny.time()
    is_after_hours = False
    if is_today:
        is_after_hours = current_time < MARKET_OPEN_TIME or current_time > MARKET_CLOSE_TIME
    
    return {
        "date": check_date.isoformat(),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_after_hours": is_after_hours if is_today else None,
        "is_open": is_market_open(session_date),
        "current_time_ny": now_ny.isoformat() if is_today else None,
    }
