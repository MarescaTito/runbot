"""
Google API utilities for BeauchBot.

Provides shared functionality for Google APIs:
- Service account authentication
- Service creation (Docs, Drive)
- Document text extraction
"""

import json
import base64
import logging
from typing import Dict, Any

# Google APIs
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Import utilities
from utils.config_utils import require_variable

logger = logging.getLogger(__name__)

# Google API scopes - comprehensive set for all BeauchBot needs
SCOPES = [
    'https://www.googleapis.com/auth/documents.readonly',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/spreadsheets'  # Full read/write access to sheets
]


def _get_service_account_credentials():
    """
    Get Google service account credentials from configuration.

    Returns:
        service_account.Credentials object

    Raises:
        ValueError: If credentials cannot be loaded
    """
    service_account_b64 = require_variable('google_service_account_b64')

    try:
        # Decode base64 and parse JSON
        service_account_json = base64.b64decode(service_account_b64).decode('utf-8')
        service_account_info = json.loads(service_account_json)

        # Create credentials
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES)

        logger.debug(f"Using service account: {service_account_info.get('client_email', 'unknown')}")

        return creds

    except Exception as e:
        raise ValueError(f"Failed to decode service account credentials: {e}")


def get_google_docs_service():
    """
    Get Google Docs service with service account authentication.
    
    Returns:
        Google Docs service object
        
    Raises:
        ValueError: If service cannot be created
    """
    try:
        creds = _get_service_account_credentials()
        return build('docs', 'v1', credentials=creds)
    except Exception as e:
        raise ValueError(f"Failed to create Google Docs service: {e}")


def get_google_drive_service():
    """
    Get Google Drive service with service account authentication.
    
    Returns:
        Google Drive service object
        
    Raises:
        ValueError: If service cannot be created
    """
    try:
        creds = _get_service_account_credentials()
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        raise ValueError(f"Failed to create Google Drive service: {e}")


def get_google_sheets_service():
    """
    Get Google Sheets service with service account authentication.
    
    Returns:
        Google Sheets service object
        
    Raises:
        ValueError: If service cannot be created
    """
    try:
        creds = _get_service_account_credentials()
        return build('sheets', 'v4', credentials=creds)
    except Exception as e:
        raise ValueError(f"Failed to create Google Sheets service: {e}")


def extract_text_from_document(doc_content: Dict[str, Any], tab_name: str | None = None) -> str:
    """
    Extract plain text from Google Docs document structure.

    Args:
        doc_content: Document content from Google Docs API
        tab_name: The tab to read from, if None read entire document

    Returns:
        Plain text content of the document
    """
    def extract_text_from_element(element):
        text = ""
        if 'textRun' in element:
            text += element['textRun'].get('content', '')
        elif 'pageBreak' in element:
            text += '\n---PAGE BREAK---\n'
        elif 'columnBreak' in element:
            text += '\n---COLUMN BREAK---\n'
        elif 'footnoteReference' in element:
            text += '[footnote]'
        elif 'horizontalRule' in element:
            text += '\n---\n'
        elif 'equation' in element:
            text += '[equation]'
        elif 'inlineObjectElement' in element:
            text += '[object]'
        return text

    def extract_text_from_paragraph(paragraph):
        text = ""
        elements = paragraph.get('elements', [])
        for element in elements:
            text += extract_text_from_element(element)
        return text

    def extract_text_from_table(table):
        text = ""
        for row in table.get('tableRows', []):
            row_text = ""
            for cell in row.get('tableCells', []):
                cell_text = ""
                for content_element in cell.get('content', []):
                    if 'paragraph' in content_element:
                        cell_text += extract_text_from_paragraph(content_element['paragraph'])
                row_text += cell_text + "\t"
            text += row_text.rstrip('\t') + "\n"
        return text

    full_text = ""
    if tab_name:
        for tab in doc_content.get('tabs'):
            tab_title = tab.get('tabProperties', {}).get('title')
            if tab_title == tab_name:
                doc_content = tab.get('documentTab', {})
                break
    body = doc_content.get('body', {})
    content_elements = body.get('content', [])
    
    for content_element in content_elements:
        if 'paragraph' in content_element:
            full_text += extract_text_from_paragraph(content_element['paragraph']) + "\n"
        elif 'table' in content_element:
            full_text += extract_text_from_table(content_element['table'])
        elif 'sectionBreak' in content_element:
            full_text += "\n---\n"  # Section break
    
    return full_text.strip()

    """
    Fetch the system prompt from a Google Doc.
    
    Reads the document ID from SYSTEM_PROMPT_DOC_ID environment variable.
    
    Returns:
        The system prompt text from the specified Google Doc
        
    Raises:
        ValueError: If SYSTEM_PROMPT_DOC_ID is not set or document cannot be accessed
    """
    system_prompt_doc_id = os.getenv('SYSTEM_PROMPT_DOC_ID')
    
    if not system_prompt_doc_id:
        raise ValueError(
            "SYSTEM_PROMPT_DOC_ID environment variable is required. "
            "Please set it to the ID of your system prompt Google Document."
        )
    
    try:
        logger.info(f"Fetching system prompt from Google Doc: {system_prompt_doc_id}")
        
        # Get Google Docs service
        docs_service = get_google_docs_service()
        
        # Retrieve the document
        document = docs_service.documents().get(documentId=system_prompt_doc_id).execute()
        
        # Extract text content
        text_content = extract_text_from_document(document)
        
        if not text_content.strip():
            raise ValueError(f"System prompt document (ID: {system_prompt_doc_id}) is empty. Please add content to the document.")
        
        logger.info("Successfully loaded system prompt from Google Doc")
        return text_content.strip()
        
    except HttpError as e:
        error_details = json.loads(e.content.decode()) if hasattr(e, 'content') else {}
        error_message = error_details.get('error', {}).get('message', 'Unknown error')
        
        if e.resp.status == 403:
            raise ValueError(
                f"Access denied to system prompt document (ID: {system_prompt_doc_id}). "
                f"Make sure the service account has permission to read the document. "
                f"Details: {error_message}"
            )
        elif e.resp.status == 404:
            raise ValueError(
                f"System prompt document not found (ID: {system_prompt_doc_id}). "
                f"Please check the SYSTEM_PROMPT_DOC_ID environment variable. "
                f"Details: {error_message}"
            )
        else:
            raise ValueError(
                f"Google API error accessing system prompt document: {error_message} "
                f"(status {e.resp.status})"
            )
    
    except ValueError:
        # Re-raise ValueError exceptions (our custom ones and Google API ones)
        raise
    
    except Exception as e:
        raise ValueError(f"Unexpected error loading system prompt: {str(e)}")