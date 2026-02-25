#!/usr/bin/env python3
"""
Ping Agent Script for BeauchBot - Windmill Compatible

This script is designed to run as a Windmill workflow.
It performs the following workflow for upcoming runs:
1. Identify and read the calendar document
2. Find runs happening within 3 hours
3. Extract BLs for each run
4. Match runs to Action Network events
5. Fetch attendees from Action Network
6. Check conversation history between BLs and attendees
7. Send group messages to attendees (with BLs included)

Windmill Variables Required (in f/run_club folder):
- openai_api_key: OpenAI API key for the LLM
- google_service_account_b64: Base64 encoded Google service account JSON
- phone_directory_doc_id: Google Doc ID containing contact information
- action_network_api_key: Action Network API key for event linking
- twilio_account_sid, twilio_auth_token, twilio_phone_number: For messaging
- attendance_sheet_id: Google Sheets ID for attendance tracking (optional, for nudges)
- allowed_bls: Comma-separated list of BL names to allow (optional, if not set all BLs are allowed)
- bl_calendar_doc_id: Google Doc ID of BL calendar (optional, if not set uses LLM to get month's calendar)
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_utils import get_variable, require_variable
from utils.twilio import get_all_messages_to_phone_number, send_text
from utils.action_network_utils import (
    fetch_all_action_network_events,
    get_event_attendees,
    match_run_to_action_network_event,
)
from utils.attendance_utils import (
    format_nudge_message,
    identify_nudge_candidates,
    parse_attendance_sheet,
)
from utils.google_utils import (
    extract_text_from_document,
    get_google_docs_service,
    get_google_drive_service,
)
from utils.phone_utils import get_allowed_contacts, normalize_phone_number

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_simulated_time(simulated_time: str) -> datetime:
    """Parse the simulated time string and return a datetime object."""
    if "," in simulated_time:
        naive_time = datetime.strptime(simulated_time, '%Y-%m-%d,%H:%M')
    else:
        naive_time = datetime.strptime(simulated_time, '%Y-%m-%d')
    return naive_time.replace(tzinfo=ZoneInfo("America/New_York"))

def identify_calendar_doc(client: OpenAI, available_docs: List[Dict[str, Any]], current_time: datetime) -> Optional[str]:
    """Use LLM to identify which document is the calendar for the current month."""
    doc_list = "\n".join([
        f"- ID: {doc['id']}, Name: {doc['name']}, Modified: {doc.get('modified_time', 'Unknown')}"
        for doc in available_docs
    ])

    current_month = current_time.strftime('%B')
    current_year = current_time.strftime('%Y')

    prompt = """You are helping identify which Google Document is the calendar/schedule document for the current month.

Current date/time: {current_datetime}
Current month: {current_month} {current_year}

Here are the available documents:
{doc_list}

Based on the document names, identify the calendar/schedule document for {current_month} {current_year}.
Look for documents with names like "{current_month} {current_year} Calendar", "{current_month} Calendar", etc.

If you cannot find an appropriate calendar document for {current_month} {current_year}, respond with exactly "NONE".
Otherwise, respond with ONLY the document ID, nothing else.""".format(
        current_datetime=current_time.strftime('%Y-%m-%d %I:%M %p %Z'),
        current_month=current_month,
        current_year=current_year,
        doc_list=doc_list
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    doc_id = response.choices[0].message.content.strip()

    if doc_id == "NONE" or not doc_id:
        logger.warning(f"No calendar document found for {current_month} {current_year}")
        return None

    logger.info(f"Identified calendar document: {doc_id}")
    return doc_id


def parse_runs_from_calendar(client: OpenAI, calendar_text: str, current_time: datetime) -> List[Dict[str, Any]]:
    """Use LLM to parse all runs from calendar document, including BL assignments."""
    current_month = current_time.strftime('%B')
    current_year = current_time.year

    prompt = """You are parsing a monthly calendar document to extract ALL runs.

Current month/year: {current_month} {current_year}

Calendar document content:
{calendar_text}

CALENDAR FORMAT:
The calendar is a monthly table/grid with days of the week as column headers (Sunday through Saturday).
Each cell contains:
1. A day number (1-31) at the top
2. Zero or more run entries, each containing:
   - Run name and time (e.g., "Office Loop 7 PM" or "Prospect Park Loop 7 PM")
   - BL (Bottom-liner) section: "BL: (H) Name" for Head, "(T) Name" for Tail
   - Additional fields like "Chat blast:", "RSVP reach out:" (ignore these)
3. Multiple runs can occur on the same day

BL (BOTTOM-LINER) INFORMATION:
- BLs are listed as "BL: (H) Name" for Head and "(T) Name" for Tail
- Some runs have both Head and Tail BLs
- Some runs have only one BL (Head or Tail)
- Some runs have no BLs assigned yet (empty or missing)
- Sometimes formatted as "BL:(H) Name" (no space after colon)
- Extract ALL BL names regardless of role (H) or (T)

TASK:
Extract ALL runs from the entire calendar month.

For each run:
1. Find the day number in the cell
2. Parse the run name and time (e.g., "Office Loop 7 PM" → name: "Office Loop", time: 7 PM)
3. Convert time to 24-hour format:
   - Times with "AM" stay as-is (7 AM = 07:00)
   - Times with "PM" add 12 (7 PM = 19:00, 7:30 PM = 19:30)
4. Combine day + month + year + time to create ISO datetime with EST timezone (-05:00)
5. Extract ALL BL names (both Head and Tail) into an array
   - If "BL: (H) Gareth (T) Nic L" → ["Gareth", "Nic L"]
   - If "BL: (H) Tim (T)" → ["Tim"]
   - If "BL: Randy" → ["Randy"]
   - If no BLs or empty → []
6. Extract the complete text for that run (including BL section)

IMPORTANT:
- Extract ALL runs from the entire month, not just upcoming ones
- Pay attention to the day number in each cell
- Parse times correctly (7 PM = 19:00, 7:30 PM = 19:30)
- Use EST timezone offset: -05:00
- Extract ALL BL names regardless of (H) or (T) designation

Respond with ONLY a JSON array. Each run should have:
- "time": ISO format datetime (e.g., "{current_year}-12-15T19:00:00-05:00" for 7 PM on Dec 15)
- "name": Run name (e.g., "Office Loop")
- "bls": Array of BL names (e.g., ["Gareth", "Nic L"] or [] if none)
- "full_text": The complete text block for this run including BL section

If no runs found, respond with [].
Example: [{{"time": "{current_year}-12-02T19:00:00-05:00", "name": "Office Loop", "bls": ["Gareth", "Nic L"], "full_text": "Office Loop 7 PM\\nBL: (H) Gareth\\n(T) Nic L"}}]""".format(
        current_month=current_month,
        current_year=current_year,
        calendar_text=calendar_text
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that parses calendar data. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    llm_response = response.choices[0].message.content.strip()

    if llm_response.startswith("```"):
        lines = llm_response.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        llm_response = '\n'.join(lines).strip()

    try:
        runs = json.loads(llm_response)
        logger.info(f"Parsed {len(runs)} total runs from calendar")
        return runs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse runs JSON: {e}")
        return []


def filter_runs_by_time_window(runs: List[Dict[str, Any]], current_time: datetime, hours: int = 3) -> List[Dict[str, Any]]:
    """Filter runs to only those occurring within the specified time window."""
    cutoff_time = current_time + timedelta(hours=hours)

    filtered_runs = []
    for run in runs:
        try:
            run_time_str = run.get('time', '')
            run_time = datetime.fromisoformat(run_time_str)

            if current_time <= run_time <= cutoff_time:
                filtered_runs.append(run)
                bl_count = len(run.get('bls', []))
                logger.info(f"Including run: {run.get('name')} at {run_time.strftime('%Y-%m-%d %I:%M %p')} ({bl_count} BLs)")
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping run with invalid time format: {run.get('name', 'Unknown')} - {e}")
            continue

    logger.info(f"Found {len(filtered_runs)} runs within {hours}-hour window")
    return filtered_runs


def filter_action_network_events_by_time_window(
    events: List[Dict[str, Any]],
    current_time: datetime,
    hours: int = 10
) -> List[Dict[str, Any]]:
    """
    Filter Action Network events to only those occurring within the specified time window.

    Args:
        events: List of Action Network events
        current_time: Current datetime
        hours: Number of hours to look ahead (default: 10)

    Returns:
        List of filtered events with parsed datetime added
    """
    eastern_tz = ZoneInfo("America/New_York")

    # Ensure current_time is in Eastern timezone
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=eastern_tz)
    else:
        current_time = current_time.astimezone(eastern_tz)

    cutoff_time = current_time + timedelta(hours=hours)

    filtered_events = []
    for event in events:
        event_start_str = event.get('start_date')

        if not event_start_str:
            continue

        try:
            # Parse Action Network datetime
            if 'T' in event_start_str:
                # Has time component
                event_start_str_clean = event_start_str.replace('Z', '')

                if '+' in event_start_str_clean or event_start_str_clean.endswith(('-00:00', '-05:00', '-04:00')):
                    # Has timezone info
                    event_start = datetime.fromisoformat(event_start_str_clean)
                else:
                    # No timezone info - assume Eastern
                    event_start = datetime.fromisoformat(event_start_str_clean)
                    event_start = event_start.replace(tzinfo=eastern_tz)

                # Convert to Eastern for comparison
                event_start = event_start.astimezone(eastern_tz)
            else:
                # Date only - treat as midnight Eastern
                event_start = datetime.fromisoformat(event_start_str)
                event_start = event_start.replace(tzinfo=eastern_tz)

            # Check if within time window
            if current_time <= event_start <= cutoff_time:
                # Add parsed datetime to event
                event_with_time = event.copy()
                event_with_time['parsed_start_time'] = event_start
                filtered_events.append(event_with_time)

                event_title = event.get('title', event.get('name', 'Unknown'))
                total_accepted = event.get('total_accepted', 0)
                logger.info(f"Including event: {event_title} at {event_start.strftime('%Y-%m-%d %I:%M %p')} ({total_accepted} RSVPs)")
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping event with invalid time format: {event.get('title', 'Unknown')} - {e}")
            continue

    logger.info(f"Found {len(filtered_events)} Action Network events within {hours}-hour window")
    return filtered_events


def match_action_network_event_to_calendar_run(
    client: OpenAI,
    event: Dict[str, Any],
    event_start_time: datetime,
    calendar_runs: List[Dict[str, Any]],
    time_window_hours: int = 12
) -> Optional[Dict[str, Any]]:
    """
    Match an Action Network event to a calendar run to find the BLs.

    Strategy:
    1. Filter calendar runs within the time window
    2. Use LLM to intelligently match based on name, time, location, and context
    3. Return matched calendar run with BLs or None

    Args:
        client: OpenAI client for LLM matching
        event: Action Network event dictionary
        event_start_time: Parsed datetime of the event
        calendar_runs: List of all calendar runs
        time_window_hours: Time window for matching (hours before/after)

    Returns:
        Matched calendar run dictionary with BLs or None if no match found
    """
    event_title = event.get('title', event.get('name', 'Unknown'))
    logger.info(f"🔍 Matching Action Network event '{event_title}' to calendar runs...")

    # Define time window
    time_start = event_start_time - timedelta(hours=time_window_hours)
    time_end = event_start_time + timedelta(hours=time_window_hours)

    logger.info(f"   Time window: {time_start.strftime('%Y-%m-%d %I:%M %p %Z')} to {time_end.strftime('%Y-%m-%d %I:%M %p %Z')}")

    # Filter calendar runs within time window
    candidates = []
    for run in calendar_runs:
        try:
            run_time_str = run.get('time', '')
            if not run_time_str:
                continue

            run_time = datetime.fromisoformat(run_time_str)

            # Check if within time window
            if time_start <= run_time <= time_end:
                time_diff_hours = abs((run_time - event_start_time).total_seconds() / 3600)

                candidates.append({
                    'run': run,
                    'run_time': run_time,
                    'time_diff_hours': time_diff_hours
                })

                bl_count = len(run.get('bls', []))
                logger.info(f"   📋 Candidate: '{run.get('name')}'")
                logger.info(f"      Time: {run_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
                logger.info(f"      Time diff: {time_diff_hours:.1f} hours")
                logger.info(f"      BLs: {bl_count}")

        except (ValueError, TypeError) as e:
            logger.warning(f"   ⚠️  Could not parse calendar run time '{run_time_str}': {e}")
            continue

    if not candidates:
        logger.info(f"   ❌ No calendar runs found within time window")
        return None

    # Use LLM to match
    logger.info(f"   🤖 Using LLM to match event to {len(candidates)} candidate(s)...")
    matched_run = _llm_match_event_to_run(client, event, event_start_time, candidates)

    if matched_run:
        logger.info(f"   ✅ MATCH FOUND: '{matched_run.get('name')}'")
        bl_count = len(matched_run.get('bls', []))
        logger.info(f"      BLs: {bl_count}")
        if matched_run.get('bls'):
            logger.info(f"      BL names: {', '.join(matched_run.get('bls', []))}")
    else:
        logger.info(f"   ❌ No suitable match found")

    return matched_run


def _llm_match_event_to_run(
    client: OpenAI,
    event: Dict[str, Any],
    event_start_time: datetime,
    candidates: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently match an Action Network event to calendar run candidates.

    Args:
        client: OpenAI client
        event: Action Network event dictionary
        event_start_time: Parsed datetime of the event
        candidates: List of candidate run dictionaries with 'run', 'run_time', 'time_diff_hours'

    Returns:
        Matched calendar run or None
    """
    # Format event information
    event_title = event.get('title', event.get('name', 'Unknown'))
    event_info = """Event Title: {event_title}
Event Time: {event_time}""".format(
        event_title=event_title,
        event_time=event_start_time.strftime('%Y-%m-%d %I:%M %p %Z')
    )

    location = event.get('location', {})
    if location:
        venue = location.get('venue')
        locality = location.get('locality')
        region = location.get('region')
        if venue or locality or region:
            loc_parts = [p for p in [venue, locality, region] if p]
            event_info += "\nLocation: {location}".format(location=', '.join(loc_parts))

    if event.get('description'):
        desc = event['description'][:200]
        event_info += "\nDescription: {desc}...".format(desc=desc)

    # Format candidates for LLM
    candidate_list = []
    for i, candidate in enumerate(candidates, 1):
        run = candidate['run']
        run_time = candidate['run_time']

        bls = run.get('bls', [])
        bl_info = "{count} BL(s): {names}".format(count=len(bls), names=', '.join(bls)) if bls else "No BLs assigned"

        run_info = """Candidate {i}:
  Run Name: {run_name}
  Run Time: {run_time}
  Time Difference: {time_diff:.1f} hours
  {bl_info}""".format(
            i=i,
            run_name=run.get('name'),
            run_time=run_time.strftime('%Y-%m-%d %I:%M %p %Z'),
            time_diff=candidate['time_diff_hours'],
            bl_info=bl_info
        )

        candidate_list.append(run_info)

    candidates_text = "\n\n".join(candidate_list)

    prompt = """You are matching an Action Network event to calendar runs to find the BL (bottom-liner) assignments.

TARGET ACTION NETWORK EVENT:
{event_info}

CANDIDATE CALENDAR RUNS:
{candidates_text}

Analyze the candidates and determine which one (if any) best matches the target event. Consider:
- Name/title similarity (events might have different names, e.g., "Office Loop Run" vs "Office Loop")
- Time proximity (runs should be close to the event time)
- Location (if available)
- Any other contextual clues

If there is a clear match, respond with ONLY the candidate number (e.g., "1", "2", etc.).
If there is no good match or the candidates are ambiguous, respond with exactly "NONE".

Your response must be a single word: either a number or "NONE".""".format(
        event_info=event_info,
        candidates_text=candidates_text
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that matches events. Always respond with a single number or 'NONE'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        llm_response = response.choices[0].message.content.strip().upper()

        logger.info(f"      LLM response: {llm_response}")

        if llm_response == "NONE":
            logger.info(f"      LLM determined no suitable match")
            return None

        # Try to parse the candidate number
        try:
            candidate_num = int(llm_response)
            if 1 <= candidate_num <= len(candidates):
                matched_candidate = candidates[candidate_num - 1]
                matched_run = matched_candidate['run']

                # Add match metadata
                matched_run['match_time_diff_hours'] = matched_candidate['time_diff_hours']
                matched_run['match_method'] = 'llm'

                logger.info(f"      LLM selected candidate {candidate_num}")
                return matched_run
            else:
                logger.warning(f"      LLM returned invalid candidate number: {candidate_num}")
                return None
        except ValueError:
            logger.warning(f"      Could not parse LLM response as number: {llm_response}")
            return None

    except Exception as e:
        logger.error(f"      ❌ Error in LLM matching: {e}")
        return None


def get_allowed_bls() -> Optional[List[str]]:
    """Get list of allowed BLs from configuration. Returns None if not set (allow all)."""
    allowed_bls_str = get_variable("allowed_bls")

    if not allowed_bls_str:
        return None

    allowed_bls = [name.strip() for name in allowed_bls_str.split(',')]
    return allowed_bls


def validate_bls_against_contacts(client: OpenAI, bl_names: List[str], contacts: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[str]]:
    """
    Validate BL names against the contact list using LLM for intelligent matching.

    Returns:
        Tuple of (valid_bl_contacts, invalid_bl_names)
    """
    if not bl_names:
        return [], []

    contact_names = [contact['name'] for contact in contacts]
    contact_list = "\n".join(["- {name}".format(name=name) for name in contact_names])
    bl_list = "\n".join(["- {name}".format(name=name) for name in bl_names])

    prompt = """You are matching names from a calendar to a contact directory.

BL names from calendar (may have variations):
{bl_list}

Valid contact names from directory:
{contact_list}

For each BL name, find the best matching contact name. Account for:
- Case differences (John vs john)
- Last name vs initial (John Smith vs John S)
- Common nicknames (Mike vs Michael, Nic vs Nick)
- Missing or extra spaces

Respond with ONLY a JSON object mapping each BL name to its best match from the contact directory.
If no good match exists for a BL name, map it to null.

Example format:
{{
  "John S": "John Smith",
  "mike": "Mike Johnson",
  "Unknown Person": null
}}""".format(
        bl_list=bl_list,
        contact_list=contact_list
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that matches names intelligently. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        llm_response = response.choices[0].message.content.strip()

        if llm_response.startswith("```"):
            lines = llm_response.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            llm_response = '\n'.join(lines).strip()

        name_mappings = json.loads(llm_response)
        contact_by_name = {contact['name']: contact for contact in contacts}

        valid_bls = []
        invalid_bls = []

        for bl_name, matched_name in name_mappings.items():
            if matched_name and matched_name in contact_by_name:
                valid_bls.append(contact_by_name[matched_name])
                if bl_name != matched_name:
                    logger.info(f"Matched '{bl_name}' → '{matched_name}'")
            else:
                invalid_bls.append(bl_name)
                logger.warning(f"No match found for '{bl_name}'")

        if invalid_bls:
            logger.warning(f"Could not match {len(invalid_bls)} BL name(s): {', '.join(invalid_bls)}")

        allowed_bls = get_allowed_bls()
        if allowed_bls:
            allowed_bls_lower = [name.lower() for name in allowed_bls]
            filtered_bls = [bl for bl in valid_bls if bl['name'].lower() in allowed_bls_lower]

            if len(filtered_bls) < len(valid_bls):
                filtered_out = [bl['name'] for bl in valid_bls if bl['name'].lower() not in allowed_bls_lower]
                logger.info(f"Filtered out {len(filtered_out)} BL(s) not in allowed list: {', '.join(filtered_out)}")

            valid_bls = filtered_bls

        logger.info(f"Validated {len(valid_bls)} BLs")
        return valid_bls, invalid_bls

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse LLM name matching response: {e}")
        logger.warning("Falling back to case-insensitive matching")

        valid_bls = []
        invalid_bls = []
        contact_by_name_lower = {contact['name'].lower(): contact for contact in contacts}

        for bl_name in bl_names:
            bl_name_lower = bl_name.lower()
            if bl_name_lower in contact_by_name_lower:
                valid_bls.append(contact_by_name_lower[bl_name_lower])
            else:
                invalid_bls.append(bl_name)
                logger.warning(f"No match found for '{bl_name}'")

        allowed_bls = get_allowed_bls()
        if allowed_bls:
            allowed_bls_lower = [name.lower() for name in allowed_bls]
            filtered_bls = [bl for bl in valid_bls if bl['name'].lower() in allowed_bls_lower]

            if len(filtered_bls) < len(valid_bls):
                filtered_out = [bl['name'] for bl in valid_bls if bl['name'].lower() not in allowed_bls_lower]
                logger.info(f"Filtered out {len(filtered_out)} BL(s) not in allowed list: {', '.join(filtered_out)}")

            valid_bls = filtered_bls

        return valid_bls, invalid_bls


def format_attendee_message(attendee_name: str, bl_names: List[str], run_name: str, run_datetime: datetime) -> str:
    """Format a personalized message to an attendee."""
    first_name = attendee_name.split()[0] if attendee_name else "there"
    bl_first_names = [bl_name.split()[0] for bl_name in bl_names]

    if len(bl_first_names) == 1:
        bl_names_str = bl_first_names[0]
    elif len(bl_first_names) == 2:
        bl_names_str = f"{bl_first_names[0]} and {bl_first_names[1]}"
    else:
        bl_names_str = f"{', '.join(bl_first_names[:-1])}, and {bl_first_names[-1]}"

    run_time_str = run_datetime.strftime('%A, %B %d at %I:%M %p')

    message = (
        f"Hello {first_name}! I've put you in a conversation with {bl_names_str} "
        f"from DSA Running Club. It looks like you are signed up for {run_name} "
        f"on {run_time_str}. Are you still planning on attending? Also if you have "
        f"any questions feel free to put them here and we'll help you out!"
    )

    return message


# Note: normalize_phone_number is now imported from utils.phone_utils
# This ensures consistent E.164 format (+1XXXXXXXXXX) throughout the application


def check_if_already_messaged_about_run(
    all_messages: List[Dict[str, Any]],
    run_name: str,
    run_datetime: datetime
) -> bool:
    """
    Deterministically check if we've already messaged about this specific run.

    Uses string matching on the predictable message format we send:
    - Attendee messages: "signed up for {run_name}" and "on {run_time_str}"
    - BL messages: "You are assigned to BL {run_name}"
    """
    if not all_messages:
        return False

    # Format the run time string exactly as it appears in our messages
    run_time_str = run_datetime.strftime('%A, %B %d at %I:%M %p')

    # Search patterns for the messages we send
    attendee_pattern_1 = f"signed up for {run_name}"
    attendee_pattern_2 = f"on {run_time_str}"
    bl_pattern = f"You are assigned to BL {run_name}"

    # Check the most recent messages (we only check the last 15)
    for msg in all_messages[:15]:
        body = msg.get('body', '')

        # Check if this is an attendee message (both patterns must be present)
        if attendee_pattern_1 in body and attendee_pattern_2 in body:
            logger.debug(f"Found attendee message match for '{run_name}' on {run_time_str}")
            return True

        # Check if this is a BL message
        if bl_pattern in body:
            logger.debug(f"Found BL message match for '{run_name}'")
            return True

    return False


def fetch_attendee_message_history(attendee_phone: str, attendee_name: str) -> List[Dict[str, Any]]:
    """
    Fetch message history for a specific attendee across all conversations.

    Args:
        attendee_phone: Phone number of the attendee (will be normalized)
        attendee_name: Name of the attendee (for logging)

    Returns:
        List of message dictionaries, or empty list on error
    """
    try:
        all_messages = get_all_messages_to_phone_number(phone_number=attendee_phone, limit=20)
        return all_messages
    except Exception as e:
        logger.warning(f"Error fetching messages for {attendee_name}: {e}")
        return []


def check_bl_message_history(valid_bl_contacts: List[Dict[str, str]], run_name: str, run_time: datetime) -> bool:
    """
    Check if we've already messaged BLs about this run. Returns True if already messaged.

    Uses deterministic string matching to check for our predictable message format.
    """
    for bl_contact in valid_bl_contacts:
        bl_name = bl_contact['name']
        bl_phone = bl_contact['phone_number']

        try:
            all_messages = get_all_messages_to_phone_number(phone_number=bl_phone, limit=20)

            if all_messages:
                already_messaged = check_if_already_messaged_about_run(
                    all_messages=all_messages,
                    run_name=run_name,
                    run_datetime=run_time
                )

                if already_messaged:
                    logger.info(f"Already messaged {bl_name} about this run")
                    return True
        except Exception as e:
            logger.warning(f"Error fetching messages for {bl_name}: {e}")

    return False


def filter_nudge_candidates_by_rsvp(
    nudge_candidates: List[Dict[str, Any]],
    attendees: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Filter out nudge candidates who are already RSVP'd, using first name matching."""
    if not attendees:
        return nudge_candidates

    rsvp_first_names_lower = set()
    for attendee in attendees:
        attendee_name = attendee.get('full_name', '')
        if attendee_name:
            first_name = attendee_name.split()[0].lower()
            rsvp_first_names_lower.add(first_name)

    nudge_candidates_filtered = []
    for candidate in nudge_candidates:
        candidate_first_name = candidate['name'].split()[0].lower()
        if candidate_first_name not in rsvp_first_names_lower:
            nudge_candidates_filtered.append(candidate)

    return nudge_candidates_filtered


def send_nudge_message_to_bls(
    bl_names: List[str],
    bl_phone_numbers: List[str],
    run_name: str,
    nudge_candidates: List[Dict[str, Any]],
    invalid_bl_names: List[str],
    dry_run: bool
) -> None:
    """Send nudge suggestions to BLs."""
    # Get attendance form link if available
    attendance_form_link = get_variable('attendance_form_link')
    nudge_message = format_nudge_message(bl_names, run_name, nudge_candidates, attendance_form_link, invalid_bl_names)

    try:
        if dry_run:
            logger.info(f"DRY RUN: Would send nudge message to BLs")
            logger.info(f"Recipients: {', '.join(bl_names)}")
            logger.info(f"Message:\n{nudge_message}")
        else:
            result = send_text(bl_phone_numbers, nudge_message)

            if "error" in result:
                logger.error(f"Failed to send nudge message to BLs: {result['error']}")
            else:
                logger.info(f"Sent nudge message to BLs")

    except Exception as e:
        logger.error(f"Error sending nudge message to BLs: {e}")


def send_messages_to_attendees(
    client: OpenAI,
    attendees: List[Dict[str, Any]],
    valid_bl_contacts: List[Dict[str, str]],
    bl_names: List[str],
    run_name: str,
    run_time: datetime,
    dry_run: bool
) -> None:
    """
    Send group messages to attendees with BLs included.

    Fetches message history on-demand for each attendee to check if they've
    already been messaged about this specific run.
    """
    bl_phone_numbers = [contact['phone_number'] for contact in valid_bl_contacts]
    # Normalize all BL phone numbers for comparison
    bl_phone_numbers_normalized = []
    for phone in bl_phone_numbers:
        try:
            bl_phone_numbers_normalized.append(normalize_phone_number(phone))
        except ValueError as e:
            logger.warning(f"Could not normalize BL phone number '{phone}': {e}")

    messages_sent = 0
    messages_failed = 0

    for attendee in attendees:
        attendee_name = attendee.get('full_name', 'Unknown')
        attendee_phone = attendee.get('primary_phone')

        if not attendee_phone:
            continue

        # Normalize attendee phone for comparison
        try:
            attendee_phone_normalized = normalize_phone_number(attendee_phone)
        except ValueError as e:
            logger.warning(f"Could not normalize attendee phone number '{attendee_phone}': {e}")
            continue

        # Skip if attendee is one of the BLs
        if attendee_phone_normalized in bl_phone_numbers_normalized:
            logger.debug(f"Skipping {attendee_name} (is a BL)")
            continue

        # Fetch message history on-demand for this specific attendee
        logger.debug(f"Fetching message history for {attendee_name}...")
        all_messages = fetch_attendee_message_history(attendee_phone_normalized, attendee_name)

        # Check if we've already messaged this attendee about this run
        already_messaged = check_if_already_messaged_about_run(
            all_messages=all_messages,
            run_name=run_name,
            run_datetime=run_time
        )

        if already_messaged:
            logger.info(f"Already messaged {attendee_name} about this run - skipping")
            continue

        # Create group message with BLs and this attendee
        group_participants = bl_phone_numbers_normalized + [attendee_phone_normalized]
        message = format_attendee_message(attendee_name, bl_names, run_name, run_time)

        try:
            if dry_run:
                logger.info(f"DRY RUN: Would send to {attendee_name}")
                messages_sent += 1
            else:
                logger.info(f"Creating group text with numbers: {', '.join(group_participants)}")
                result = send_text(group_participants, message)

                if "error" in result:
                    logger.error(f"Failed to send to {attendee_name}: {result['error']}")
                    messages_failed += 1
                else:
                    logger.info(f"Sent to {attendee_name}")
                    messages_sent += 1

        except Exception as e:
            logger.error(f"Error sending to {attendee_name}: {e}")
            messages_failed += 1

    logger.info(f"Messaging complete: {messages_sent} sent, {messages_failed} failed")


def process_action_network_event(
    event: Dict[str, Any],
    client: OpenAI,
    contacts: List[Dict[str, str]],
    calendar_runs: List[Dict[str, Any]],
    attendance_data: List[Dict[str, Any]],
    current_time: datetime,
    include_nudges: bool,
    dry_run: bool
) -> None:
    """
    Process a single Action Network event: match to calendar run to find BLs, fetch attendees, send messages.

    This is the main processing function for the Action Network-first workflow.
    """
    event_title = event.get('title', event.get('name', 'Unknown'))
    event_start_time = event.get('parsed_start_time')
    total_accepted = event.get('total_accepted', 0)

    logger.info(f"\nProcessing Action Network event: {event_title}")
    logger.info(f"Time: {event_start_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
    logger.info(f"RSVPs: {total_accepted}")

    # Match to calendar run to find BLs
    matched_run = match_action_network_event_to_calendar_run(
        client=client,
        event=event,
        event_start_time=event_start_time,
        calendar_runs=calendar_runs,
        time_window_hours=12
    )

    if not matched_run:
        logger.warning(f"No calendar run found for event '{event_title}' - skipping")
        return

    run_name = matched_run.get('name', 'Unknown')
    bl_names = matched_run.get('bls', [])
    run_time = datetime.fromisoformat(matched_run.get('time', ''))

    if not bl_names:
        logger.warning(f"No BLs assigned for matched run '{run_name}' - skipping")
        return

    logger.info(f"Found {len(bl_names)} BLs: {', '.join(bl_names)}")

    # Validate BL contacts
    valid_bl_contacts, invalid_bl_names = validate_bls_against_contacts(client, bl_names, contacts)

    if not valid_bl_contacts:
        logger.warning(f"No valid BL contacts for run '{run_name}' - skipping")
        return

    # Fetch attendees from Action Network
    attendees = []
    try:
        event_id = event.get('identifiers', [None])[0]
        if event_id:
            event_id = event_id.split(':')[-1]
            attendees = get_event_attendees(event_id, max_attendances=100)
            logger.info(f"Fetched {len(attendees)} attendees from Action Network")
        else:
            logger.warning(f"No event ID found for event '{event_title}'")
    except Exception as e:
        logger.warning(f"Error fetching attendees: {e}")

    if not attendees:
        logger.info(f"No attendees found for event '{event_title}' - skipping messaging")
        return

    # Check if we should skip BL message (already sent)
    skip_bl_message = False
    if include_nudges and valid_bl_contacts:
        logger.info(f"Checking BL message history...")
        skip_bl_message = check_bl_message_history(valid_bl_contacts, run_name, run_time)

    # Identify nudge candidates
    nudge_candidates = []
    if include_nudges and attendance_data and not skip_bl_message:
        day_of_week = run_time.strftime('%A')

        # Pass both Action Network event name and calendar run name for LLM matching
        target_run_names = [event_title, run_name]
        logger.info(f"Analyzing attendance for nudge candidates (matching against '{event_title}' or '{run_name}')...")
        all_nudge_candidates = identify_nudge_candidates(
            target_run_names=target_run_names,
            target_day_of_week=day_of_week,
            current_time=current_time,
            all_runs=attendance_data,
            client=client
        )

        # Filter out BLs using fuzzy matching (handles name variations)
        nudge_candidates = []
        for candidate in all_nudge_candidates:
            is_bl = False
            candidate_name = candidate['name'].lower()

            # Check if candidate matches any BL name with fuzzy matching
            from difflib import SequenceMatcher
            for bl_name in bl_names:
                bl_name_lower = bl_name.lower()
                # Calculate similarity
                similarity = SequenceMatcher(None, candidate_name, bl_name_lower).ratio()

                # High threshold for BL exclusion (0.8) - must be very similar to exclude
                if similarity >= 0.8:
                    is_bl = True
                    logger.info(f"   Excluding '{candidate['name']}' from nudges (matches BL '{bl_name}' with {similarity:.2f} similarity)")
                    break

            if not is_bl:
                nudge_candidates.append(candidate)

        nudge_candidates = filter_nudge_candidates_by_rsvp(nudge_candidates, attendees)

        logger.info(f"Identified {len(nudge_candidates)} nudge candidates")

    # Send nudge suggestions to BLs
    if include_nudges and valid_bl_contacts and not skip_bl_message:
        logger.info(f"Sending nudge suggestions to BLs...")
        # Extract validated BL names from valid contacts
        validated_bl_names = [contact['name'] for contact in valid_bl_contacts]
        bl_phone_numbers = [contact['phone_number'] for contact in valid_bl_contacts]

        send_nudge_message_to_bls(validated_bl_names, bl_phone_numbers, run_name, nudge_candidates, invalid_bl_names, dry_run)

    # Send messages to attendees
    if attendees and valid_bl_contacts:
        logger.info(f"Sending messages to attendees...")
        # Extract validated BL names from valid contacts
        validated_bl_names = [contact['name'] for contact in valid_bl_contacts]
        send_messages_to_attendees(
            client, attendees, valid_bl_contacts, validated_bl_names, run_name, run_time, dry_run
        )


def run_cron_execution(simulated_time: Optional[str] = None, dry_run: bool = False, include_nudges: bool = False) -> int:
    """Execute the cron job workflow."""
    start_time = datetime.now()
    logger.info(f"Starting BeauchBot cron execution at {start_time}")

    try:
        # Check for required configuration variables
        required_vars = ["openai_api_key", "google_service_account_b64", "phone_directory_doc_id"]
        missing_vars = []
        for var in required_vars:
            if not get_variable(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required configuration variables: {', '.join(missing_vars)}")
            logger.error(f"Set as Windmill variables (f/run_club/<name>) or environment variables (UPPER_CASE)")
            return 1

        eastern_tz = ZoneInfo("America/New_York")

        if simulated_time:
            try:
                current_time = parse_simulated_time(simulated_time)
                logger.info(f"Using simulated time: {current_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
            except ValueError:
                logger.error(f"Invalid simulated time format: {simulated_time}")
                return 1
        else:
            current_time = datetime.now(eastern_tz)
            logger.info(f"Current time: {current_time.strftime('%Y-%m-%d %I:%M %p %Z')}")

        client = OpenAI(api_key=require_variable('openai_api_key'))

        drive_service = get_google_drive_service()
        results = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=20,
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc"
        ).execute()

        available_docs = results.get('files', [])
        if not available_docs:
            logger.error("No Google Docs found")
            return 1

        calendar_doc_id_from_env = os.getenv("BL_CALENDAR_DOC_ID")
        using_consolidated_doc = calendar_doc_id_from_env in available_docs
        calendar_doc_id = calendar_doc_id_from_env if using_consolidated_doc else identify_calendar_doc(client, available_docs, current_time)
        if not calendar_doc_id:
            current_month = current_time.strftime('%B')
            current_year = current_time.strftime('%Y')
            logger.error(f"Could not find calendar document for {current_month} {current_year}")
            return 1

        docs_service = get_google_docs_service()
        document = docs_service.documents().get(documentId=calendar_doc_id, includeTabsContent=using_consolidated_doc).execute()
        tab = current_time.strftime('%B') if using_consolidated_doc else None

        calendar_text = extract_text_from_document(document, tab)

        # Fetch Action Network events first
        action_network_events = []
        try:
            action_network_events = fetch_all_action_network_events(max_pages=3)
            logger.info(f"Loaded {len(action_network_events)} Action Network events")
        except Exception as e:
            logger.error(f"Failed to fetch Action Network events: {e}")
            logger.warning("Cannot continue without Action Network events")
            return 1

        # Filter Action Network events by time window
        applicable_hours = 10
        filtered_events = filter_action_network_events_by_time_window(
            action_network_events, current_time, hours=applicable_hours
        )

        if not filtered_events:
            logger.info(f"No Action Network events found within {applicable_hours} hours")
            return 0

        # Parse ALL calendar runs (not filtered by time) so we can match against them
        all_calendar_runs = parse_runs_from_calendar(client, calendar_text, current_time)

        if not all_calendar_runs:
            logger.warning("No runs found in calendar")
            return 0

        logger.info(f"Parsed {len(all_calendar_runs)} total runs from calendar")

        # Load attendance data for nudge suggestions
        attendance_data = []
        if include_nudges:
            try:
                attendance_data = parse_attendance_sheet()
                logger.info(f"Loaded attendance data for {len(attendance_data)} runs")
            except Exception as e:
                logger.error(f"Failed to load attendance data: {e}")
                logger.warning("Continuing without nudge suggestions")

        # Load contacts
        contacts = get_allowed_contacts()
        if not contacts:
            logger.error("Could not load contacts from phone directory")
            return 1

        logger.info(f"Loaded {len(contacts)} contacts from phone directory")

        # Process each Action Network event
        for i, event in enumerate(filtered_events, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Event {i}/{len(filtered_events)}")
            process_action_network_event(
                event, client, contacts, all_calendar_runs, attendance_data,
                current_time, include_nudges, dry_run
            )

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Cron execution completed successfully")
        logger.info(f"Duration: {duration}")

        return 0

    except Exception as e:
        logger.error(f"Cron execution failed: {e}", exc_info=True)
        return 1


def main(
    dry_run: bool = False,
    simulate_time: str = "",
    include_nudges: bool = False
):
    """
    Main entry point for Windmill workflow.

    Args:
        dry_run: If True, don't actually send messages (default: False)
        simulate_time: Simulate time in format 'YYYY-MM-DD,HH:MM' or 'YYYY-MM-DD' (default: current time)
        include_nudges: If True, include attendance-based nudge analysis (default: False)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    eastern_tz = ZoneInfo("America/New_York")
    now = parse_simulated_time(simulate_time) if simulate_time else datetime.now(eastern_tz)

    if now.hour < 8 or now.hour >= 20:
        logger.info(f"Outside operating hours (8 AM - 8 PM). Current hour: {now.hour}")
        return 0

    exit_code = run_cron_execution(
        simulated_time=simulate_time if simulate_time else None,
        dry_run=dry_run,
        include_nudges=include_nudges
    )

    if exit_code == 0:
        logger.info("Cron job completed successfully")
    else:
        logger.error("Cron job failed")

    return exit_code


if __name__ == "__main__":
    # For local testing, you can still run this script
    import argparse
    parser = argparse.ArgumentParser(description="BeauchBot cron job entry point")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run (don't send texts)")
    parser.add_argument("--simulate-time", "-t", help="Simulate time (format: 'YYYY-MM-DD,HH:MM' or 'YYYY-MM-DD')", default="")
    parser.add_argument("--include-nudges", "-n", action="store_true", help="Include nudge suggestions (default: disabled)")
    args = parser.parse_args()

    sys.exit(main(
        dry_run=args.dry_run,
        simulate_time=args.simulate_time,
        include_nudges=args.include_nudges
    ))
