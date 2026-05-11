#!/bin/bash
# Generate a larger Job-Shop input file for benchmarking
# Usage: ./generate_benchmark_input.sh <jobs> <machines> <output_file>

JOBS=${1:-20}
MACHINES=${2:-20}
OUTPUT=${3:-data/benchmark_large.jss}

if [ -f "$OUTPUT" ]; then
    read -p "File $OUTPUT exists. Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo "Generating $JOBS x $MACHINES job-shop input..."

{
    echo "$JOBS $MACHINES"
    for ((j = 0; j < JOBS; j++)); do
        for ((m = 0; m < MACHINES; m++)); do
            machine=$((RANDOM % MACHINES))
            duration=$((3 + RANDOM % 12))
            printf "%d %d" "$machine" "$duration"
            if [ $m -lt $((MACHINES - 1)) ]; then printf " "; fi
        done
        echo
    done
} > "$OUTPUT"

echo "Generated: $OUTPUT"
echo "Size: $(wc -l < "$OUTPUT") lines"
echo ""
echo "To benchmark this input, run:"
echo "  ./benchmark.sh 5 $OUTPUT"
