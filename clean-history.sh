#!/bin/bash
# Create a file with replacement patterns
cat > replacements.txt << EOF
regex:sk-ant-api03-[a-zA-Z0-9]{48}=ANTHROPIC_API_KEY_PLACEHOLDER
EOF

# Run git filter-repo with the replacements file
git filter-repo --force --replace-text replacements.txt
