"""
Timeframe utilities for time-bounded searches.

This module provides an enum for common timeframes and utilities
to convert them to datetime bounds.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum


class Timeframe(str, Enum):
    """
    Common timeframes for filtering search results.

    Each timeframe maps to a specific date range relative to the current time.
    """

    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"


def _get_today_start() -> datetime:
    """Get the start of today (midnight)."""
    now = datetime.now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_bounds_today() -> tuple[datetime, datetime]:
    """Get bounds for today."""
    start = _get_today_start()
    return start, start + timedelta(days=1)


def _get_bounds_yesterday() -> tuple[datetime, datetime]:
    """Get bounds for yesterday."""
    today_start = _get_today_start()
    return today_start - timedelta(days=1), today_start


def _get_bounds_this_week() -> tuple[datetime, datetime]:
    """Get bounds for this week (starting Monday)."""
    now = datetime.now()
    today_start = _get_today_start()
    days_since_monday = now.weekday()
    start = today_start - timedelta(days=days_since_monday)
    return start, now + timedelta(seconds=1)


def _get_bounds_last_week() -> tuple[datetime, datetime]:
    """Get bounds for last week."""
    now = datetime.now()
    today_start = _get_today_start()
    days_since_monday = now.weekday()
    this_monday = today_start - timedelta(days=days_since_monday)
    return this_monday - timedelta(days=7), this_monday


def _get_bounds_this_month() -> tuple[datetime, datetime]:
    """Get bounds for this month."""
    now = datetime.now()
    today_start = _get_today_start()
    return today_start.replace(day=1), now + timedelta(seconds=1)


def _get_bounds_last_month() -> tuple[datetime, datetime]:
    """Get bounds for last month."""
    today_start = _get_today_start()
    first_of_this_month = today_start.replace(day=1)
    last_month = first_of_this_month - timedelta(days=1)
    return last_month.replace(day=1), first_of_this_month


def _get_bounds_last_n_days(days: int) -> tuple[datetime, datetime]:
    """Get bounds for the last N days."""
    now = datetime.now()
    today_start = _get_today_start()
    return today_start - timedelta(days=days), now + timedelta(seconds=1)


def _get_bounds_this_year() -> tuple[datetime, datetime]:
    """Get bounds for this year."""
    now = datetime.now()
    today_start = _get_today_start()
    return today_start.replace(month=1, day=1), now + timedelta(seconds=1)


def _get_bounds_last_year() -> tuple[datetime, datetime]:
    """Get bounds for last year."""
    today_start = _get_today_start()
    this_year_start = today_start.replace(month=1, day=1)
    return this_year_start.replace(year=this_year_start.year - 1), this_year_start


# Mapping of timeframes to their bound calculation functions
_TIMEFRAME_BOUNDS: dict[Timeframe, Callable[[], tuple[datetime, datetime]]] = {
    Timeframe.TODAY: _get_bounds_today,
    Timeframe.YESTERDAY: _get_bounds_yesterday,
    Timeframe.THIS_WEEK: _get_bounds_this_week,
    Timeframe.LAST_WEEK: _get_bounds_last_week,
    Timeframe.THIS_MONTH: _get_bounds_this_month,
    Timeframe.LAST_MONTH: _get_bounds_last_month,
    Timeframe.LAST_7_DAYS: lambda: _get_bounds_last_n_days(7),
    Timeframe.LAST_30_DAYS: lambda: _get_bounds_last_n_days(30),
    Timeframe.LAST_90_DAYS: lambda: _get_bounds_last_n_days(90),
    Timeframe.THIS_YEAR: _get_bounds_this_year,
    Timeframe.LAST_YEAR: _get_bounds_last_year,
}


def get_timeframe_bounds(timeframe: Timeframe) -> tuple[datetime, datetime]:
    """
    Convert a timeframe enum to start and end datetime bounds.

    Args:
        timeframe: The timeframe to convert.

    Returns:
        Tuple of (start_datetime, end_datetime) representing the bounds.
        Start is inclusive, end is exclusive.

    Example:
        >>> start, end = get_timeframe_bounds(Timeframe.TODAY)
        >>> # start = today at 00:00:00, end = tomorrow at 00:00:00
    """
    bounds_fn = _TIMEFRAME_BOUNDS.get(timeframe)
    if bounds_fn:
        return bounds_fn()

    # Default fallback: very wide range
    now = datetime.now()
    return datetime(1970, 1, 1), now + timedelta(days=365 * 100)


def parse_date_string(date_str: str) -> datetime | None:
    """
    Parse a date string in various formats.

    Args:
        date_str: Date string in ISO format or common formats.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None
