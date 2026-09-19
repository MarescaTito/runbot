"""
Action Network API utilities for BeauchBot.

Provides read-only functionality to:
- Fetch events from Action Network
- Match calendar runs to Action Network events
- Query event details by name, date, and location
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import utilities
from f.running_club.config_utils import require_variable
from f.running_club.phone_utils import normalize_phone_number

# Action Network API base URL
ACTION_NETWORK_API_BASE = "https://actionnetwork.org/api"


def get_action_network_headers() -> Dict[str, str]:
    """
    Get headers for Action Network API requests.

    Returns:
        Dictionary with required headers including API token

    Raises:
        ValueError: If action_network_api_key is not set
    """
    api_key = require_variable('action_network_api_key')

    return {
        'OSDI-API-Token': api_key
    }


def fetch_action_network_events(page: int = 1, per_page: int = 25) -> Dict[str, Any]:
    """
    Fetch events from Action Network API.

    Args:
        page: Page number for pagination (default: 1)
        per_page: Number of results per page (default: 25)

    Returns:
        API response dictionary with events and pagination info

    Raises:
        requests.RequestException: If API request fails
    """
    url = f"{ACTION_NETWORK_API_BASE}/v2/events"
    headers = get_action_network_headers()

    params = {
        'page': page,
        'per_page': per_page
    }

    logger.info(f"📡 Fetching Action Network events (page {page}, per_page {per_page})...")
    logger.info(f"   Request URL: {url}")
    logger.info(f"   Request headers: {', '.join(headers.keys())}")
    # Log API key length for debugging (without exposing the key)
    try:
        from utils.config_utils import get_variable
        api_key = get_variable('action_network_api_key')
        api_key_len = len(api_key) if api_key else 0
    except:
        api_key_len = 0
    logger.info(f"   API key length: {api_key_len} characters")

    try:
        response = requests.get(url, headers=headers, params=params)

        logger.info(f"   Response status: {response.status_code}")
        logger.info(f"   Response headers: {dict(response.headers)}")

        if response.status_code == 403:
            logger.error(f"❌ 403 Forbidden - API key may be invalid or lacks permissions")
            logger.error(f"   Response body: {response.text[:500]}")
            raise requests.exceptions.HTTPError(f"403 Forbidden: {response.text[:200]}")

        response.raise_for_status()

        data = response.json()

        # Log pagination info
        total_pages = data.get('total_pages', 0)
        total_records = data.get('total_records', 0)
        logger.info(f"✅ Fetched page {page}/{total_pages} ({total_records} total events)")

        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP Error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request Error: {e}")
        raise


def fetch_all_action_network_events(max_pages: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch all events from Action Network API (up to max_pages).

    Args:
        max_pages: Maximum number of pages to fetch (default: 10)

    Returns:
        List of event dictionaries
    """
    all_events = []
    page = 1

    logger.info(f"📡 Fetching all Action Network events (max {max_pages} pages)...")

    while page <= max_pages:
        try:
            previous_len = len(all_events)
            data = fetch_action_network_events(page=page)

            # Extract events from the embedded data
            events = data.get('_embedded', {}).get('osdi:events', [])

            if not events:
                logger.info(f"✅ No more events on page {page}, stopping")
                break

            for event in events:
                if not event.get("status") == "cancelled":
                    all_events.append(event)
                else:
                    logger.info("Skipping cancelled event " + event.get('title'))

            logger.info(f"   Added {len(all_events) - previous_len} events from page {page}")

            # Check if there are more pages
            total_pages = data.get('total_pages', 0)
            if page >= total_pages:
                logger.info(f"✅ Reached last page ({page}/{total_pages})")
                break

            page += 1

        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch page {page}: {e}")
            break

    logger.info(f"✅ Fetched {len(all_events)} total events from Action Network")
    return all_events


def extract_event_details(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key details from an Action Network event object.

    Args:
        event: Raw event object from Action Network API

    Returns:
        Dictionary with extracted details:
        - id: Event ID
        - title: Event title
        - name: Internal event name
        - start_date: Start datetime
        - location: Location info (venue, address, coordinates)
    """
    return {
        'id': event.get('identifiers', [None])[0] if event.get('identifiers') else None,
        'title': event.get('title', ''),
        'name': event.get('name', ''),
        'description': event.get('description', ''),
        'start_date': event.get('start_date'),
        'end_date': event.get('end_date'),
        'location': event.get('location', {}),
        'status': event.get('status'),
        'total_accepted': event.get('total_accepted', 0),
        '_links': event.get('_links', {})
    }


def match_run_to_action_network_event(
    run_name: str,
    run_datetime: datetime,
    action_network_events: List[Dict[str, Any]],
    openai_client=None,
    time_window_hours: int = 24
) -> Optional[Dict[str, Any]]:
    """
    Match a calendar run to an Action Network event using LLM-based matching.

    Strategy:
    1. Filter events within the time window
    2. Use LLM to intelligently match based on name, time, location, and context
    3. Return matched event or None

    Note: Both calendar runs and Action Network events are treated as EST/EDT.

    Args:
        run_name: Name of the run from calendar
        run_datetime: Datetime of the run (should be in EST/EDT)
        action_network_events: List of Action Network events
        openai_client: OpenAI client for LLM matching (required)
        time_window_hours: Time window for matching (hours before/after)

    Returns:
        Matched event details or None if no match found

    Raises:
        ValueError: If openai_client is not provided
        RuntimeError: If LLM matching fails
    """
    from zoneinfo import ZoneInfo

    eastern_tz = ZoneInfo("America/New_York")

    # Ensure run_datetime is in Eastern timezone
    if run_datetime.tzinfo is None:
        # If naive, assume it's already Eastern time
        run_datetime = run_datetime.replace(tzinfo=eastern_tz)
    else:
        # Convert to Eastern if it's in a different timezone
        run_datetime = run_datetime.astimezone(eastern_tz)

    logger.info(f"🔍 Matching run '{run_name}' at {run_datetime.strftime('%Y-%m-%d %I:%M %p %Z')} to Action Network...")

    if not action_network_events:
        logger.warning("⚠️  No Action Network events to match against")
        return None

    # Define time window
    time_start = run_datetime - timedelta(hours=time_window_hours)
    time_end = run_datetime + timedelta(hours=time_window_hours)

    logger.info(f"   Time window: {time_start.strftime('%Y-%m-%d %I:%M %p %Z')} to {time_end.strftime('%Y-%m-%d %I:%M %p %Z')}")

    # Filter events within time window
    candidates = []
    for event in action_network_events:
        event_start_str = event.get('start_date')

        if not event_start_str:
            continue

        try:
            # Parse Action Network datetime
            # According to docs, start_date is in "local timezone" but ISO format
            # We'll treat it as Eastern timezone for consistency
            if 'T' in event_start_str:
                # Has time component - parse as ISO format
                # Remove 'Z' if present and parse
                event_start_str_clean = event_start_str.replace('Z', '')

                # Parse the datetime
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
            if time_start <= event_start <= time_end:
                event_details = extract_event_details(event)

                event_title = event_details['title'] or event_details['name']
                time_diff_hours = abs((event_start - run_datetime).total_seconds() / 3600)

                candidates.append({
                    'event': event_details,
                    'event_start': event_start,
                    'time_diff_hours': time_diff_hours
                })

                logger.info(f"   📋 Candidate: '{event_title}'")
                logger.info(f"      Start: {event_start.strftime('%Y-%m-%d %I:%M %p %Z')}")
                logger.info(f"      Time diff: {time_diff_hours:.1f} hours")

                location = event_details.get('location', {})
                if location:
                    venue = location.get('venue')
                    locality = location.get('locality')
                    if venue or locality:
                        logger.info(f"      Location: {venue or ''} {locality or ''}".strip())
            else:
                # Log events outside the window at debug level
                event_title = event.get('title') or event.get('name', 'Unknown')
                logger.debug(f"   ⏭️  Skipped '{event_title}': outside time window")
                logger.debug(f"      Start: {event_start.strftime('%Y-%m-%d %I:%M %p %Z')}")
                logger.debug(f"      Window: {time_start.strftime('%Y-%m-%d %I:%M %p %Z')} to {time_end.strftime('%Y-%m-%d %I:%M %p %Z')}")

        except (ValueError, TypeError) as e:
            logger.warning(f"   ⚠️  Could not parse event date '{event_start_str}': {e}")
            continue

    if not candidates:
        logger.info(f"   ❌ No Action Network events found within time window")
        return None

    # Use LLM to match (required)
    if not openai_client:
        raise ValueError("OpenAI client is required for event matching")

    logger.info(f"   🤖 Using LLM to match run to {len(candidates)} candidate(s)...")
    matched_event = _llm_match_event(openai_client, run_name, run_datetime, candidates)

    if matched_event:
        event_title = matched_event['title'] or matched_event['name']
        logger.info(f"   ✅ MATCH FOUND: '{event_title}'")
        logger.info(f"      Event ID: {matched_event['id']}")
        logger.info(f"      Time diff: {matched_event.get('match_time_diff_hours', 0):.1f} hours")
    else:
        logger.info(f"   ❌ No suitable match found")

    return matched_event


def _llm_match_event(openai_client, run_name: str, run_datetime: datetime, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently match a run to candidate Action Network events.

    Args:
        openai_client: OpenAI client
        run_name: Name of the run from calendar
        run_datetime: Datetime of the run
        candidates: List of candidate event dictionaries with 'event', 'event_start', 'time_diff_hours'

    Returns:
        Matched event details or None
    """
    import json

    # Format run information
    run_info = """Run Name: {run_name}
Run Time: {run_time}""".format(
        run_name=run_name,
        run_time=run_datetime.strftime('%Y-%m-%d %I:%M %p %Z')
    )

    # Format candidates for LLM
    candidate_list = []
    for i, candidate in enumerate(candidates, 1):
        event = candidate['event']
        event_start = candidate['event_start']

        event_info = """Candidate {i}:
  Title: {title}
  Start Time: {start_time}
  Time Difference: {time_diff:.1f} hours
  Event ID: {event_id}""".format(
            i=i,
            title=event['title'] or event['name'],
            start_time=event_start.strftime('%Y-%m-%d %I:%M %p %Z'),
            time_diff=candidate['time_diff_hours'],
            event_id=event['id']
        )

        location = event.get('location', {})
        if location:
            venue = location.get('venue')
            locality = location.get('locality')
            region = location.get('region')
            if venue or locality or region:
                loc_parts = [p for p in [venue, locality, region] if p]
                event_info += "\n  Location: {location}".format(location=', '.join(loc_parts))

        if event.get('description'):
            # Truncate description to avoid token limits
            desc = event['description'][:200]
            event_info += "\n  Description: {desc}...".format(desc=desc)

        candidate_list.append(event_info)

    candidates_text = "\n\n".join(candidate_list)

    prompt = """You are matching a calendar run event to Action Network events.

TARGET RUN:
{run_info}

CANDIDATE ACTION NETWORK EVENTS:
{candidates_text}

Analyze the candidates and determine which one (if any) best matches the target run. Consider:
- Name/title similarity (runs might have different names than events, e.g., "Office Loop" vs "Office Loop Run")
- Time proximity (events should be close to the run time)
- Location (if available)
- Any other contextual clues

If there is a clear match, respond with ONLY the candidate number (e.g., "1", "2", etc.).
If there is no good match or the candidates are ambiguous, respond with exactly "NONE".

Your response must be a single word: either a number or "NONE".""".format(
        run_info=run_info,
        candidates_text=candidates_text
    )

    try:
        response = openai_client.chat.completions.create(
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
                matched_event = matched_candidate['event']

                # Add match metadata
                matched_event['match_time_diff_hours'] = matched_candidate['time_diff_hours']
                matched_event['match_method'] = 'llm'

                logger.info(f"      LLM selected candidate {candidate_num}")
                return matched_event
            else:
                logger.warning(f"      LLM returned invalid candidate number: {candidate_num}")
                return None
        except ValueError:
            logger.warning(f"      Could not parse LLM response as number: {llm_response}")
            return None

    except Exception as e:
        logger.error(f"      ❌ Error in LLM matching: {e}")
        raise RuntimeError(f"LLM matching failed: {e}") from e


def fetch_event_attendances(event_id: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch all attendances for a specific Action Network event.

    Args:
        event_id: Action Network event ID
        max_pages: Maximum number of pages to fetch (default: 10)

    Returns:
        List of attendance records
    """
    all_attendances = []
    page = 1
    headers = get_action_network_headers()

    logger.info(f"📡 Fetching attendances for event {event_id}...")

    while page <= max_pages:
        url = f"{ACTION_NETWORK_API_BASE}/v1/events/{event_id}/attendance"
        params = {'page': page, 'per_page': 25}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            # Extract attendances from embedded data
            attendances = data['_embedded']['osdi:attendance']

            if not attendances:
                logger.info(f"   No more attendances on page {page}")
                break

            all_attendances.extend(attendances)
            logger.info(f"   Fetched {len(attendances)} attendances from page {page}")

            # Check if there are more pages
            total_pages = data.get('total_pages', 0)
            if page >= total_pages:
                break

            page += 1

        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch attendances page {page}: {e}")
            break

    logger.info(f"✅ Fetched {len(all_attendances)} total attendances")
    return all_attendances


def fetch_person_details(person_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch details for a specific person from Action Network.

    Args:
        person_id: Action Network person ID

    Returns:
        Dictionary with person details (name, email, phone) or None if not found
    """
    url = f"{ACTION_NETWORK_API_BASE}/v2/people/{person_id}"
    headers = get_action_network_headers()

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        person = response.json()

        # Extract key fields
        person_details = {
            'id': person_id,
            'given_name': person.get('given_name', ''),
            'family_name': person.get('family_name', ''),
            'email_addresses': person.get('email_addresses', []),
            'phone_numbers': person.get('phone_numbers', []),
        }

        # Get primary email
        primary_email = None
        for email in person_details['email_addresses']:
            if email.get('primary'):
                primary_email = email.get('address')
                break
        if not primary_email and person_details['email_addresses']:
            primary_email = person_details['email_addresses'][0].get('address')

        person_details['primary_email'] = primary_email

        # Get primary phone and normalize it
        primary_phone = None
        for phone in person_details['phone_numbers']:
            if phone.get('primary'):
                primary_phone = phone.get('number')
                break
        if not primary_phone and person_details['phone_numbers']:
            primary_phone = person_details['phone_numbers'][0].get('number')

        # Normalize phone number to E.164 format (+1XXXXXXXXXX)
        if primary_phone:
            try:
                primary_phone = normalize_phone_number(primary_phone)
            except ValueError as e:
                logger.warning(f"Could not normalize phone number '{primary_phone}' for person {person_id}: {e}")

        person_details['primary_phone'] = primary_phone

        # Full name
        person_details['full_name'] = f"{person_details['given_name']} {person_details['family_name']}".strip()

        return person_details

    except requests.RequestException as e:
        logger.warning(f"⚠️  Could not fetch person {person_id}: {e}")
        return None


def get_event_attendees(event_id: str, max_attendances: int = 250) -> List[Dict[str, Any]]:
    """
    Get all attendees (people) for an Action Network event with full details.

    Args:
        event_id: Action Network event ID
        max_attendances: Maximum number of attendances to fetch (default: 250)

    Returns:
        List of attendee dictionaries with name, email, phone
    """
    logger.info(f"👥 Fetching attendees for event {event_id}...")

    # Fetch attendances
    attendances = fetch_event_attendances(event_id, max_pages=(max_attendances // 25) + 1)

    if not attendances:
        logger.info(f"   No attendances found")
        return []

    # Extract unique person IDs
    person_ids = []
    for attendance in attendances[:max_attendances]:
        person_id = attendance.get('action_network:person_id')
        if person_id and person_id not in person_ids:
            person_ids.append(person_id)

    logger.info(f"   Found {len(person_ids)} unique attendees")

    # Fetch person details for each attendee
    attendees = []
    for i, person_id in enumerate(person_ids, 1):
        logger.debug(f"   Fetching person {i}/{len(person_ids)}: {person_id}")

        person_details = fetch_person_details(person_id)
        if person_details:
            attendees.append(person_details)

            # Log attendee info
            name = person_details.get('full_name', 'Unknown')

            logger.info(f"      {i}. {name}")

    logger.info(f"✅ Retrieved {len(attendees)} attendee details")
    return attendees
