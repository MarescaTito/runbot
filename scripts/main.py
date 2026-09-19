#!/usr/bin/env python3

#requirements:
# google-api-python-client>=2.147.0
# google-auth>=2.35.0
# requests>=2.31.0
# twilio>=9.0.0
# backoff>=2.2.1
# wmill

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from f.running_club.config_utils import get_variable, require_variable
from f.running_club.twilio import get_all_messages_to_phone_number, send_text
from f.running_club.action_network_utils import (
    fetch_all_action_network_events,
    get_event_attendees,
    match_run_to_action_network_event,
)
from f.running_club.google_utils import (
    extract_text_from_document,
    get_google_docs_service,
    get_google_drive_service,
)
from f.running_club.calendar_utils import parse_calendar, get_bls_from_calendar
from f.running_club.discord_utils import send_discord_messages
from f.running_club.phone_utils import normalize_phone_number


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


def validate_bls_against_contacts(bl_names: List[str], contacts: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[str]]:
    """
    Validate BL names against the contact list deterministically assuming exact matches.

    Returns:
        Tuple of (valid_bl_contacts, invalid_bl_names)
    """
    if not bl_names:
        return [], []

    contact_by_name = {
        contact["name"].strip().lower(): contact
        for contact in contacts
        if contact.get("name")
    }

    valid_bls = []
    invalid_bls = []
    for bl_name in bl_names:
        normalized = bl_name.strip().lower()

        matched_contact = contact_by_name.get(normalized)

        if matched_contact:
            valid_bls.append(matched_contact)
        else:
            invalid_bls.append(bl_name)
            logger.warning(f"No match found for '{bl_name}'")

    if invalid_bls:
        logger.warning(
            f"Could not match {len(invalid_bls)} BL name(s): {', '.join(invalid_bls)}"
        )

    logger.info(f"Validated {len(valid_bls)} BLs")

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
        f"Hi {first_name}, ComradeBot here. You’re signed up for {run_name} on {run_time_str}. "
        f"I've connected you with my human comrade(s), {bl_names_str}, "
        f"from DSA Running Club who can help with any questions. "
        f"Are you still planning to attend? "
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

    # Check the most recent messages (we only check the last 15)
    for msg in all_messages[:15]:
        body = msg.get('body', '')

        if run_name.lower() in body.lower() and run_time_str.lower() in body.lower():
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


def send_messages_to_attendees(
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
    contacts: List[Dict[str, str]],
    calendar_runs: List[Dict[str, Any]],
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

    matched_run = None
    for run in calendar_runs:
        if run.get('name') == event_title.strip():
            matched_run = run
            break

    if not matched_run:
        logger.warning(f"No calendar run found for event '{event_title}' - skipping")
        return

    run_name = matched_run.get('name', 'Unknown')
    bl_names = matched_run.get('bls', [])
    run_time = event.get('parsed_start_time')

    if not bl_names:
        logger.warning(f"No BLs assigned for matched run '{run_name}' - skipping")
        return
    logger.info(f"Found {len(bl_names)} BLs: {', '.join(bl_names)}")

    valid_bl_contacts, invalid_bl_names = validate_bls_against_contacts(bl_names, contacts)
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

    # Send messages to attendees
    if attendees and valid_bl_contacts:
        logger.info(f"Sending messages to attendees...")
        # Extract validated BL names from valid contacts
        validated_bl_names = [contact['name'] for contact in valid_bl_contacts]
        send_messages_to_attendees(
            attendees, valid_bl_contacts, validated_bl_names, run_name, run_time, dry_run
        )


def run_cron_execution(simulated_time: Optional[str] = None, dry_run: bool = False) -> int:
    """Execute the cron job workflow."""
    start_time = datetime.now()
    logger.info(f"Starting BeauchBot cron execution at {start_time}")

    try:
        # Check for required configuration variables
        required_vars = ["google_service_account_b64", "phone_directory_doc_id"]
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

        current_month = current_time.strftime('%B')

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
        applicable_hours = int(require_variable("lookback_hours"))
        filtered_events = filter_action_network_events_by_time_window(
            action_network_events, current_time, hours=applicable_hours
        )

        if not filtered_events:
            logger.info(f"No Action Network events found within {applicable_hours} hours")
            return 0

        next_month = current_time.replace(month=current_time.month + 1)
        if current_time.month == 12:
            next_month = current_time.replace(year=current_time.year + 1, month=1)

        all_calendar_runs = parse_calendar(current_month) + parse_calendar(next_month.strftime('%B'))

        if current_time.hour == 8:
            filtered_events_for_discord = filter_action_network_events_by_time_window(
                action_network_events, current_time, hours=(24 * 8)
            )
            logger.info(f"Checking {len(filtered_events_for_discord)} runs for BLs for Discord")
            send_discord_messages(filtered_events_for_discord, all_calendar_runs)

        if not all_calendar_runs:
            logger.warning("No runs found in calendar")
            return 0

        logger.info(f"Parsed {len(all_calendar_runs)} total runs from calendar")

        # Load contacts
        contacts = get_bls_from_calendar()
        if not contacts:
            logger.error("Could not load contacts from phone directory")
            return 1

        logger.info(f"Loaded {len(contacts)} contacts from phone directory")

        # Process each Action Network event
        for i, event in enumerate(filtered_events, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Event {i}/{len(filtered_events)}")
            process_action_network_event(
                event, contacts, all_calendar_runs, dry_run
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

    if now.hour < 8 or now.hour > 20:
        logger.info(f"Outside operating hours (8 AM - 8 PM). Current hour: {now.hour}")
        return 0

    exit_code = run_cron_execution(
        simulated_time=simulate_time if simulate_time else None,
        dry_run=dry_run,
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
