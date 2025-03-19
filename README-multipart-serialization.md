# Multipart Message Serialization

This document explains how to work with the MultipartMessage serialization formats in the Fast Agent library.

## Overview

The serialization system provides two primary formats for converting PromptMessageMultipart objects to and from persistent storage:

1. **JSON Format** - Complete serialization with full fidelity
2. **Delimited Text Format** - Human-readable with plain text and JSON hybrid

These formats allow you to save, load, and exchange multipart messages containing text, resources, and images.

## JSON Format

The JSON format provides a complete serialization of all multipart message properties, making it ideal for:

- Programmatic storage and retrieval
- API communication
- Preserving full data fidelity

### Usage

```python
from mcp_agent.mcp.prompt_serialization import (
    multipart_messages_to_json, 
    json_to_multipart_messages,
    save_messages_to_json_file,
    load_messages_from_json_file
)

# Serialize to JSON string
json_str = multipart_messages_to_json(messages)

# Deserialize from JSON string
messages = json_to_multipart_messages(json_str)

# Save to file
save_messages_to_json_file(messages, "conversation.json")

# Load from file
messages = load_messages_from_json_file("conversation.json")
```

## Delimited Text Format

The delimited text format uses a hybrid approach:
- Plain text with delimiters for USER and ASSISTANT roles
- JSON for resources and images

This format is ideal for:
- Human readability and editing
- Maintaining simplicity for text content
- Preserving structure for complex content like resources and images

### Format Structure

```
---USER
This is user plain text content

---RESOURCE
{
  "type": "resource",
  "resource": {
    "uri": "resource://example.py",
    "mimeType": "text/x-python",
    "text": "print('Hello, world!')"
  }
}

---ASSISTANT
This is assistant plain text content

---RESOURCE
{
  "type": "image",
  "url": "https://example.com/image.png",
  "mimeType": "image/png",
  "data": "base64EncodedData"
}
```

### Usage

```python
from mcp_agent.mcp.prompt_serialization import (
    multipart_messages_to_delimited_format,
    delimited_format_to_multipart_messages,
    save_messages_to_delimited_file,
    load_messages_from_delimited_file
)

# Convert to delimited format
delimited_content = multipart_messages_to_delimited_format(messages)
delimited_text = "\n".join(delimited_content)

# Parse from delimited format
messages = delimited_format_to_multipart_messages(delimited_text)

# Save to file
save_messages_to_delimited_file(messages, "conversation.txt")

# Load from file
messages = load_messages_from_delimited_file("conversation.txt")
```

## Working with Resources

Resources are serialized as JSON in both formats. The JSON representation preserves:

- URI
- MIME type
- Text content
- Blob data (if present)

Example resource JSON:

```json
{
  "type": "resource",
  "resource": {
    "uri": "resource://example.py",
    "mimeType": "text/x-python",
    "text": "def hello():\n    print('Hello, world!')"
  }
}
```

## Working with Images

Images are also serialized as JSON in both formats:

```json
{
  "type": "image",
  "data": "base64EncodedImageData",
  "mimeType": "image/png",
  "url": "https://example.com/image.png"
}
```

## Implementation Details

### AnyUrl Handling

The serialization handles Pydantic's `AnyUrl` type correctly using the `model_dump(mode="json")` approach, ensuring proper serialization of URI properties.

### Text Content Combining

In the delimited format, multiple TextContent objects for the same role are combined with double newlines (`\n\n`) between them. This improves readability in the serialized format while preserving paragraph breaks.

### Round-Trip Conversion

Both formats support round-trip conversion, allowing you to serialize and deserialize without data loss for all standard content types.

## Choosing a Format

- Use **JSON format** when:
  - Working programmatically with the full data structure
  - Storing in databases or APIs
  - You need to preserve exact structure and metadata

- Use **Delimited format** when:
  - Human readability of text content is important
  - Users need to manually edit the content
  - The file will be stored in version control

## Best Practices

1. Prefer the JSON format for programmatic storage and retrieval
2. Use the delimited format for human-editable files
3. When deserializing from unknown sources, try both formats
4. For maximum compatibility, include a format identifier in your files