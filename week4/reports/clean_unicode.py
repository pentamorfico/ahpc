#!/usr/bin/env python3
"""Clean Unicode characters from markdown for LaTeX compatibility"""

import re

# Read the file
with open('AHPC_Assignment4_Report.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Dictionary of Unicode replacements
replacements = {
    '↔': '<->',
    '→': '->',
    '←': '<-',
    '×': 'x',
    '≥': '>=',
    '≤': '<=',
    '≈': '~=',
    '≠': '!=',
    'μ': 'u',
    '°': ' deg',
    '–': '-',
    '—': '-',
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    '…': '...',
    '•': '*',
    '✓': 'OK',
    '✅': '[OK]',
    '⭐': '*',
    '🚀': '',
    '🎉': '',
    '📊': '',
    '📄': '',
    '⏳': '',
    '⚠️': 'WARNING',
    '❌': '[X]',
}

# Apply replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Remove any remaining non-ASCII characters (except basic punctuation)
# Keep only ASCII printable characters and common punctuation
content = ''.join(char if ord(char) < 128 else ' ' for char in content)

# Write cleaned version
with open('AHPC_Assignment4_Report_clean.md', 'w', encoding='ascii') as f:
    f.write(content)

print("✓ Cleaned version created: AHPC_Assignment4_Report_clean.md")
