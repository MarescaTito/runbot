"""
Calendar utilities for BeauchBot.

Provides functionality to parse calendar data from Google Sheets
"""

import os
import wmill
import logging
from typing import List, Dict, Any

from f.running_club.config_utils import require_variable

logger = logging.getLogger(__name__)

def parse_calendar(month: str) ->List[Dict[str, Any]]:
    from f.running_club.google_utils import get_google_sheets_service

    sheet_id = require_variable('sheet_calendar')
    logger.info("Fetching calendar sheet...")

    sheets_service = get_google_sheets_service()
    result = sheets_service.spreadsheets().values().get(spreadsheetId = sheet_id, range= month + "!B2:D100").execute()
    values = result.get('values', [])

    if not values or len(values) < 2:
        logger.warning("⚠️  Calendar sheet has insufficient data")
        return []

    runs = []

    logger.info(f"Processing {len(values)} runs...")

    for i, row in enumerate(values):
        if len(row) < 3:
            continue

        head = row[0].strip()
        tail = row[1].strip()
        if not head and not tail:
            # logger.warning(f"⚠️  Row {i}: Missing bottom liners, skipping")
            continue

        an_event = row[2].strip()
        if not an_event:
            # logger.warning(f"⚠️  Row {i}: Missing AN event, skipping")
            continue

        runs.append({"bls": [head, tail], "name": an_event})

    logger.info(f"✅ Parsed {len(runs)} runs from calendar")

    return runs

def get_bls_from_calendar() -> List[Dict[str, str]]:
    from f.running_club.google_utils import get_google_sheets_service

    sheet_id = require_variable('sheet_calendar')
    logger.info("Fetching calendar sheet...")

    sheets_service = get_google_sheets_service()
    result = sheets_service.spreadsheets().values().get(spreadsheetId = sheet_id, range= "BLs" + "!A1:B40").execute()
    values = result.get('values', [])

    contacts = []

    for row in values:
        if len(row) < 2:
            continue

        contacts.append({
            'name': row[0].strip(),
            'phone_number': row[1].strip()
        })

    return contacts