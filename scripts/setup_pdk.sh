#!/bin/bash
set -e

# Define where the PDK will live
export PDK_ROOT="/app/sky130_volare"
export PDK_VERSION="bdc9412b3e468c102d01b7cf6337be06ec6e9c9a"

echo "---------------------------------------------------"
echo "Initializing Sky130 PDK Setup"
echo "Target Directory: $PDK_ROOT"
echo "Target Version:   $PDK_VERSION"
echo "---------------------------------------------------"

# 1. Check if the critical model file already exists
# If it does, we assume the PDK is installed and skip the heavy lifting.
EXPECTED_LIB="$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice"

if [ -f "$EXPECTED_LIB" ]; then
    echo "PDK models found at $EXPECTED_LIB"
    echo "Skipping download/install."
    exit 0
fi

# 2. Ensure volare is installed
if ! command -v volare &> /dev/null; then
    echo "Volare not found in path. Attempting pip install..."
    pip install volare
fi

# 3. Enable (Download & Install) the PDK
echo "PDK not found. Starting Volare download..."
volare enable --pdk sky130 --pdk-root "$PDK_ROOT" "$PDK_VERSION"

echo "---------------------------------------------------"
echo "PDK Setup Complete."
echo "Models are located at: $EXPECTED_LIB"
echo "---------------------------------------------------"
