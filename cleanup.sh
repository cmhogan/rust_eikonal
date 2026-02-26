#!/usr/bin/env bash
#
# Remove generated/intermediate files that are not needed in the repository.
# Keeps: source code, configs, docs, and the 4 PNG files referenced by README.md.
#
set -euo pipefail

cd "$(dirname "$0")"

echo "Cleaning up generated files..."

# Cargo build artifacts
if [ -d target ]; then
    echo "  Removing target/"
    rm -rf target
fi

# Clean out the ignored demo_output directory
if [ -d demo_output ]; then
    echo "  Cleaning demo_output/"
    rm -rf demo_output/*
fi

# Generated reference data in test_fixtures (keep the Python script)
for f in test_fixtures/*.npy; do
    [ -e "$f" ] || continue
    echo "  Removing $f"
    rm "$f"
done

# Stray .improvements file (review artifact)
if [ -f improvements.md ]; then
    echo "  Removing improvements.md"
    rm improvements.md
fi

echo "Done."
