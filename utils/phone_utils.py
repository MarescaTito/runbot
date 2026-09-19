"""
Phone number utilities for BeauchBot.

Provides shared functionality for:
- Phone number parsing and validation
- Contact management
- Phone number normalization

All phone numbers in this project are normalized to E.164 format: +1XXXXXXXXXX
where +1 is the US country code and X represents 10 digits.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def normalize_phone_number(phone: str) -> str:
    """
    Normalize a phone number to E.164 format (+1XXXXXXXXXX).

    This is the canonical phone number format used throughout the BeauchBot project.
    All phone numbers are assumed to be US numbers (country code +1).

    Handles various input formats:
    - With/without country code: "5551234567", "15551234567", "+15551234567"
    - With separators: "(555) 123-4567", "555.123.4567", "555-123-4567"
    - With spaces: "555 123 4567", "+1 555 123 4567"

    Args:
        phone: Phone number in any format

    Returns:
        Phone number in E.164 format (+1XXXXXXXXXX)

    Raises:
        ValueError: If phone number cannot be normalized to valid US format
    """
    if not phone:
        raise ValueError("Phone number cannot be empty")

    # Remove all non-digit characters except leading +
    cleaned = re.sub(r'[^\d+]', '', phone)

    # Remove + sign temporarily to work with just digits
    if cleaned.startswith('+'):
        cleaned = cleaned[1:]

    # Remove any remaining + signs that aren't at the start
    cleaned = cleaned.replace('+', '')

    # Ensure we have only digits
    if not cleaned.isdigit():
        raise ValueError(f"Phone number contains invalid characters: {phone}")

    # Handle different digit counts
    if len(cleaned) == 10:
        # 10 digits - assume US number, add +1
        return f"+1{cleaned}"
    elif len(cleaned) == 11 and cleaned.startswith('1'):
        # 11 digits starting with 1 - US number with country code
        return f"+{cleaned}"
    elif len(cleaned) == 11 and not cleaned.startswith('1'):
        raise ValueError(f"11-digit number must start with 1 (country code): {phone}")
    else:
        raise ValueError(f"Invalid phone number length ({len(cleaned)} digits): {phone}. Expected 10 or 11 digits.")


def validate_phone_number(phone: str) -> bool:
    """
    Validate if a phone number is in the correct E.164 format (+1XXXXXXXXXX).

    Args:
        phone: Phone number to validate

    Returns:
        True if phone number is in valid E.164 format, False otherwise
    """
    try:
        normalized = normalize_phone_number(phone)
        # Check if it matches the expected pattern
        return bool(re.match(r'^\+1\d{10}$', normalized))
    except (ValueError, TypeError):
        return False


def parse_phone_numbers_from_text(text_content: str) -> List[Dict[str, str]]:
    """
    Parse phone numbers and names from text content.

    Expected format: "Name: Phone Number" on each line
    This function is resilient to formatting variations like:
    - Extra whitespace
    - Different separators (: or -)
    - Various phone number formats
    - Empty lines

    Args:
        text_content: Raw text content from the document

    Returns:
        List of dictionaries with 'name' and 'phone_number' keys
    """
    contacts = []

    # Split by lines and process each line
    lines = text_content.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip lines that look like headers or section dividers
        if line.startswith('#') or line.startswith('---') or line.lower() in ['phone directory', 'contacts', 'numbers']:
            continue

        # Try to match patterns like "Name: Phone" or "Name - Phone"
        # This regex looks for: characters before separator, then separator (: or - variants), then phone content
        pattern = r'^([^:–—-]+?)[\s]*[:–—-][\s]*(.+)$'
        match = re.match(pattern, line)

        if match:
            name_part = match.group(1).strip()
            phone_part = match.group(2).strip()

            # Basic validation - name should have at least 2 characters
            if len(name_part) < 2:
                continue

            # Basic phone validation - should contain digits
            if not re.search(r'\d', phone_part):
                continue

            # Normalize phone number to E.164 format (+1XXXXXXXXXX)
            try:
                formatted_phone = normalize_phone_number(phone_part)
            except ValueError as e:
                # Log warning but keep the contact with original phone format for debugging
                logger.warning(f"Could not normalize phone number '{phone_part}' for '{name_part}': {e}")
                formatted_phone = phone_part

            contacts.append({
                'name': name_part,
                'phone_number': formatted_phone,
                'original_line': line  # Keep original for debugging
            })
        else:
            # Log lines that don't match expected format for debugging
            logger.debug(f"Could not parse line: '{line}'")

    logger.info(f"Parsed {len(contacts)} contacts from phone directory")
    return contacts


def format_contact_list_for_error(contacts: List[Dict[str, str]]) -> str:
    """
    Format a list of contacts for display in error messages.
    
    Args:
        contacts: List of contact dictionaries
        
    Returns:
        Formatted string showing available contacts
    """
    if not contacts:
        return "No contacts available. Please check the phone directory document."
    
    # Sort contacts by name for better presentation
    sorted_contacts = sorted(contacts, key=lambda c: c['name'].lower())
    
    contact_lines = []
    for contact in sorted_contacts:
        contact_lines.append(f"  - {contact['name']}: {contact['phone_number']}")
    
    return f"Available contacts:\n" + "\n".join(contact_lines)
