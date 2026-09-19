import wmill
import logging
import requests
from typing import List, Dict, Any, Tuple

from f.running_club.config_utils import require_variable


def send_discord_messages(events: List[Dict[str, Any]], calendar_runs: List[Dict[str, Any]]) -> None:
    matched_events: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unmatched_events: List[str] = []

    for event in events:
        event_title = event.get('title', event.get('name', 'Unknown'))
        matched = False
        for calendar_run in calendar_runs:
            if calendar_run.get('name') == event_title.strip():
                matched_events.append((event, calendar_run))
                matched = True
                break
        if not matched:
            unmatched_events.append(event_title)

    logging.info(f"Found {len(matched_events)} events with calendar records")
    logging.info(f"Found {len(unmatched_events)} events with no calendar records")

    for event in unmatched_events:
        send_discord_message(f"‼️‼️ ALERT: {event} does not match any bottom-lined runs ‼️‼️")

    for (event, calendar_run) in matched_events:
        event_title = event.get('title', event.get('name', 'Unknown'))
        if not calendar_run.get('bls'):
            send_discord_message(f"‼️‼️ ALERT: {event_title} does not have any BLs ‼️‼️")


def send_discord_message(message: str):
    try:
        webhook_url = require_variable("discord_url")
        data = {
            "content": message,
            "username": "ComradeBot"
        }

        result = requests.post(webhook_url, json=data)

        if result.status_code == 204:
            print("Success")
        else:
            logging.error(f"Failed to send discord message with code {result.status_code}: {result.text}")
    except Exception as e:
        logging.error(f"Failed to send discord message code {e}")
