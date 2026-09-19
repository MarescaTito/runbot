"""
Twilio tools for BeauchBot.

Provides functionality to:
- Send individual SMS and Group MMS messages
- Get conversation history for individuals and groups
- Get contact phone numbers

All phone numbers are normalized to E.164 format (+1XXXXXXXXXX) for consistency.
"""

import logging
import re
from typing import List, Dict, Any

# Twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Import shared utilities
from f.running_club.config_utils import require_variable
from f.running_club.phone_utils import normalize_phone_number

logger = logging.getLogger(__name__)

# Reduce Twilio logging verbosity
logging.getLogger('twilio').setLevel(logging.WARNING)
logging.getLogger('twilio.http_client').setLevel(logging.WARNING)

def get_twilio_client():
    """Initialize and return a Twilio client using configuration variables."""
    account_sid = require_variable('twilio_account_sid')
    auth_token = require_variable('twilio_auth_token')

    return Client(account_sid, auth_token)


def get_twilio_phone_number() -> str:
    """
    Get the Twilio phone number from configuration.

    Returns:
        Normalized phone number in E.164 format (+1XXXXXXXXXX)

    Raises:
        ValueError: If configuration variable is not set or phone number is invalid
    """
    twilio_number = require_variable('twilio_phone_number')

    try:
        return normalize_phone_number(twilio_number)
    except ValueError as e:
        raise ValueError(f"Invalid twilio_phone_number: {e}")


def get_my_phone_number() -> str:
    """
    Get my personal phone number from configuration.

    Returns:
        Normalized phone number in E.164 format (+1XXXXXXXXXX)

    Raises:
        ValueError: If configuration variable is not set or phone number is invalid
    """
    my_number = require_variable('my_phone_number')

    try:
        return normalize_phone_number(my_number)
    except ValueError as e:
        raise ValueError(f"Invalid my_phone_number: {e}")


def send_text(to_numbers: List[str], message: str) -> Dict[str, Any]:
    """Send a text message to an individual or group via Twilio.

    For individual messaging (1 number): Standard SMS between you and one recipient
    For group messaging (2+ numbers): Creates Group MMS where all participants see each other's messages
    Group MMS requires US/Canada (+1) numbers and creates true group conversations.
    Automatically reuses existing conversations with the same participants to avoid conflicts.

    Args:
        to_numbers: List of phone numbers in any format (will be normalized to E.164: +1XXXXXXXXXX)
        message: The message content to send

    Returns:
        Message status information or group conversation details including 'reused_existing' flag
    """
    try:
        if not to_numbers or len(to_numbers) == 0:
            return {"error": "At least one phone number is required"}

        # Normalize all phone numbers to E.164 format
        normalized_numbers = []
        for phone in to_numbers:
            try:
                normalized = normalize_phone_number(phone)
                normalized_numbers.append(normalized)
            except ValueError as e:
                return {"error": f"Invalid phone number '{phone}': {e}"}

        client = get_twilio_client()
        from_number = get_twilio_phone_number()

        # Determine if this is individual or group messaging based on recipient count
        if len(normalized_numbers) == 1:
            # Individual messaging using standard SMS
            return _send_individual_text(client, from_number, normalized_numbers[0], message)
        else:
            # Group messaging using Group MMS
            return _send_group_text(client, from_number, normalized_numbers, message)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"error": str(e)}

    except Exception as e:
        logger.error(f"Error sending text: {e}")
        return {"error": f"Failed to send text: {str(e)}"}

# ============================================================================
# HELPER FUNCTIONS (Internal)
# ============================================================================

def _send_individual_text(client, from_number: str, to_number: str, message: str) -> Dict[str, Any]:
    """Send individual SMS message."""

    # Send SMS using standard Twilio messaging
    message_result = client.messages.create(
        body=message,
        from_=from_number,
        to=to_number
    )

    response = {
        "type": "individual",
        "message_sid": message_result.sid,
        "to": to_number,
        "from": from_number,
        "body": message,
        "status": message_result.status,
        "date_created": message_result.date_created.isoformat() if message_result.date_created else None
    }

    logger.info(f"Individual SMS sent to {to_number}, Message: {message_result.sid}")
    return response


def _send_group_text(client, from_number: str, to_numbers: List[str], message: str) -> Dict[str, Any]:
    """Send Group MMS message using Conversations API, handling duplicates natively."""
    try:
        # Validate US/Canada numbers (Group MMS requirement)
        for phone_number in to_numbers:
            if not phone_number.startswith('+1'):
                return {"error": f"Group MMS only supports US/Canada (+1) numbers. Invalid: {phone_number}"}

        # Inline helper to handle sending to an verified existing conversation SID
        def send_to_existing(conversation_sid: str) -> Dict[str, Any]:
            # Ensure beauchbot_assistant participant exists in the conversation
            participants = client.conversations.v1.conversations(conversation_sid).participants.list()
            beauchbot_participant_exists = any(p.identity == "beauchbot_assistant" for p in participants)

            if not beauchbot_participant_exists:
                logger.info(f"Adding beauchbot_assistant participant to existing conversation {conversation_sid}")
                client.conversations.v1.conversations(conversation_sid).participants.create(
                    identity="beauchbot_assistant",
                    messaging_binding_projected_address=from_number
                )

            message_result = client.conversations.v1.conversations(conversation_sid).messages.create(
                body=message,
                author="beauchbot_assistant"
            )
            return {
                "type": "group",
                "conversation_sid": conversation_sid,
                "message_sid": message_result.sid,
                "reused_existing": True,
                "body": message,
                "date_created": message_result.date_created.isoformat() if message_result.date_created else None
            }

        # --- OPTIMISTIC CREATION FLOW ---
        # Instead of an expensive scan loop, we try building the room immediately.
        logger.info(f"Initiating Group MMS conversation creation candidate with {len(to_numbers)} participants")
        conversation = client.conversations.v1.conversations.create(
            friendly_name=f"Group conversation {len(to_numbers)} participants"
        )

        # Add SMS participants
        for to_number in to_numbers:
            client.conversations.v1.conversations(conversation.sid).participants.create(
                messaging_binding_address=to_number
            )

        # Add business chat participant (This triggers Twilio's Group MMS validation)
        try:
            client.conversations.v1.conversations(conversation.sid).participants.create(
                identity="beauchbot_assistant",
                messaging_binding_projected_address=from_number
            )
        except TwilioRestException as e:
            # Check if this configuration already exists somewhere else
            if e.status == 409 and "already exists as Conversation" in str(e):
                # Regex match for a standard Twilio Conversation SID (starts with CH followed by 32 hex chars)
                match = re.search(r'CH[a-f0-9]{32}', str(e))
                if match:
                    existing_sid = match.group(0)
                    logger.info(f"Twilio 409 intercepted. Existing conversation mapped to: {existing_sid}")

                    # Clean up the broken shell conversation we just initialized
                    try:
                        conversation.delete()
                    except Exception as delete_err:
                        logger.warning(f"Failed to clean up temporary stub {conversation.sid}: {delete_err}")

                    # Seamlessly pivot and send via the pre-existing conversation room
                    return send_to_existing(existing_sid)

            # If it's a different variety of Twilio exception, bubble it up
            raise e

        # If the creation succeeded perfectly without conflicts, deliver the message here
        message_result = client.conversations.v1.conversations(conversation.sid).messages.create(
            body=message,
            author="beauchbot_assistant"
        )

        logger.info(f"Group MMS sent successfully to new conversation: {conversation.sid}")
        return {
            "type": "group",
            "conversation_sid": conversation.sid,
            "message_sid": message_result.sid,
            "reused_existing": False,
            "body": message,
            "date_created": message_result.date_created.isoformat() if message_result.date_created else None
        }

    except Exception as e:
        logger.error(f"Error sending group text: {e}")
        return {"error": f"Failed to send group text: {str(e)}"}


def get_all_messages_to_phone_number(phone_number: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get all messages sent from our Twilio number to a specific phone number.

    This searches across all conversations (individual and group) to find any messages
    we've sent to this person, regardless of which conversation they were in.

    Args:
        phone_number: Phone number in any format (will be normalized to E.164: +1XXXXXXXXXX)
        limit: Maximum number of messages to retrieve (default: 20)

    Returns:
        List of message dictionaries with body, date_created, and conversation info
    """
    try:
        # Normalize the phone number
        try:
            phone_number = normalize_phone_number(phone_number)
        except ValueError as e:
            logger.error(f"Invalid phone number '{phone_number}': {e}")
            return []

        client = get_twilio_client()
        twilio_number = get_twilio_phone_number()

        all_messages = []

        # Search through recent conversations to find ones that include this phone number
        conversations = client.conversations.v1.conversations.list(state='active')

        for conversation in conversations:
            try:
                # Get participants for this conversation
                participants = client.conversations.v1.conversations(conversation.sid).participants.list()

                # Check if the target phone number is a participant
                participant_found = False
                for participant in participants:
                    if participant.messaging_binding:
                        participant_address = None
                        if hasattr(participant.messaging_binding, 'address') and participant.messaging_binding.address:
                            participant_address = participant.messaging_binding.address
                        elif isinstance(participant.messaging_binding, dict) and participant.messaging_binding.get('address'):
                            participant_address = participant.messaging_binding['address']

                        if participant_address == phone_number:
                            participant_found = True
                            break

                # If this conversation includes the target phone number, get messages
                if participant_found:
                    # Get messages from this conversation sent by beauchbot_assistant
                    messages = client.conversations.v1.conversations(conversation.sid).messages.list(
                        limit=limit,
                        order='desc'
                    )

                    for msg in messages:
                        # Only include messages sent by us (beauchbot_assistant)
                        if msg.author == "beauchbot_assistant":
                            all_messages.append({
                                "conversation_sid": conversation.sid,
                                "body": msg.body,
                                "date_created": msg.date_created.isoformat() if msg.date_created else None,
                                "message_sid": msg.sid
                            })

            except Exception as e:
                logger.warning(f"Error checking conversation {conversation.sid}: {e}")
                continue

        # Sort by date (most recent first)
        all_messages.sort(key=lambda x: x.get('date_created', ''), reverse=True)

        # Limit to requested number
        all_messages = all_messages[:limit]

        logger.debug(f"Found {len(all_messages)} messages sent to {phone_number} across all conversations")
        return all_messages

    except Exception as e:
        logger.error(f"Error fetching all messages to {phone_number}: {e}")
        return []
