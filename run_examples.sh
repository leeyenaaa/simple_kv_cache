#!/bin/bash
# run_examples.sh - Execute all KV cache eviction examples
#
# This script demonstrates the complete workflow of KV cache eviction strategies:
# 1. Basic token selection test (test_simple_evict.py)
# 2. LongBench evaluation with Sink+Recent strategy
# 3. LongBench evaluation with Sink+Recent+Uniform strategy
# 4. Comparison of both strategies
#
# Usage:
#   ./run_examples.sh
#
# The script will:
# - Activate the LaCache virtual environment
# - Create an output directory for results
# - Run all examples in sequence with clear progress indicators
# - Save JSON results for comparison
# - Display a final summary
#
# Requirements:
# - LaCache virtual environment at ../LaCache/.venv
# - Python 3.8+
# - All dependencies installed (see requirements.txt)

set -e  # Exit immediately on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Print header
echo ""
echo "======================================================================"
echo "KV Cache Eviction - Complete Example Workflow"
echo "======================================================================"
echo ""

# Step 1: Activate virtual environment
echo -e "${BLUE}[1/5]${NC} Activating LaCache virtual environment..."
if [ ! -d "../LaCache/.venv" ]; then
    echo -e "${RED}Error: LaCache virtual environment not found at ../LaCache/.venv${NC}"
    echo "Please ensure LaCache is installed at ../LaCache/"
    exit 1
fi

source ../LaCache/.venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 2: Create output directory
echo -e "${BLUE}[2/5]${NC} Creating output directory..."
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory created: $OUTPUT_DIR${NC}"
echo ""

# Step 3: Run basic eviction test
echo -e "${BLUE}[3/5]${NC} Running basic token selection test..."
echo "======================================================================" 
python test_simple_evict.py
echo "======================================================================" 
echo -e "${GREEN}✓ Basic test completed${NC}"
echo ""

# Step 4: Run LongBench evaluation with Strategy 1 (Sink+Recent)
echo -e "${BLUE}[4/5]${NC} Running LongBench evaluation (Strategy 1: Sink+Recent)..."
echo "This may take a minute or two..."
STRATEGY1_OUTPUT="$OUTPUT_DIR/strategy1_sink_recent.json"
python run_longbench_example.py \
    --demo_mode \
    --strategy sink_recent \
    --output "$STRATEGY1_OUTPUT"
echo -e "${GREEN}✓ Strategy 1 evaluation completed${NC}"
echo "  Results saved to: $STRATEGY1_OUTPUT"
echo ""

# Step 5: Run LongBench evaluation with Strategy 2 (Sink+Recent+Uniform)
echo -e "${BLUE}[5/5]${NC} Running LongBench evaluation (Strategy 2: Sink+Recent+Uniform)..."
echo "This may take a minute or two..."
STRATEGY2_OUTPUT="$OUTPUT_DIR/strategy2_sink_recent_uniform.json"
python run_longbench_example.py \
    --demo_mode \
    --strategy sink_recent_uniform \
    --middle_budget 256 \
    --output "$STRATEGY2_OUTPUT"
echo -e "${GREEN}✓ Strategy 2 evaluation completed${NC}"
echo "  Results saved to: $STRATEGY2_OUTPUT"
echo ""

# Step 6: Compare strategies
echo -e "${BLUE}[6/6]${NC} Comparing eviction strategies..."
echo "======================================================================" 
python compare_strategies.py "$STRATEGY1_OUTPUT" "$STRATEGY2_OUTPUT"
echo "======================================================================" 
echo -e "${GREEN}✓ Strategy comparison completed${NC}"
echo ""

# Final summary
echo "======================================================================"
echo -e "${GREEN}✓ Workflow Complete!${NC}"
echo "======================================================================"
echo ""
echo "Summary of Results:"
echo "  - Basic test output: Displayed above"
echo "  - Strategy 1 results: $STRATEGY1_OUTPUT"
echo "  - Strategy 2 results: $STRATEGY2_OUTPUT"
echo "  - Comparison output: Displayed above"
echo ""
echo "Next Steps:"
echo "  1. Review the comparison output above to see performance differences"
echo "  2. Examine JSON files for detailed metrics and timing information"
echo "  3. Run with --plot flag for visualizations:"
echo "     python compare_strategies.py $STRATEGY1_OUTPUT $STRATEGY2_OUTPUT --plot"
echo ""
echo "For more information, see README.md"
echo ""
