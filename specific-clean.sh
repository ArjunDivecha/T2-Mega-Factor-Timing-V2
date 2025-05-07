#!/bin/bash
# Create a file with a very specific replacement pattern
cat > specific-replacements.txt << EOF
# Replace the specific API key pattern that GitHub is detecting
regex:sk-ant-api03-[a-zA-Z0-9]{24,60}=ANTHROPIC_API_KEY_PLACEHOLDER
# Also try with quotes around it
regex:"sk-ant-api03-[a-zA-Z0-9]{24,60}"="ANTHROPIC_API_KEY_PLACEHOLDER"
# And with single quotes
regex:'sk-ant-api03-[a-zA-Z0-9]{24,60}'='ANTHROPIC_API_KEY_PLACEHOLDER'
EOF

# Run git filter-repo with the specific replacements file
git filter-repo --force --replace-text specific-replacements.txt
