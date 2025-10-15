#!/bin/bash



chunk_size=(1000 1000 1000 1000 2000 2000 2000 3000)
chunk_index=(0 1 2 3 2 6 5 7)
epochs=(4 4 4 4 4 4 4 4)

for i in "${!chunk_size[@]}"; do
  echo "Running ${epochs[$i]} epochs with chunk size: ${chunk_size[$i]} and chunk index: ${chunk_index[$i]}..."
  python llm.py "${chunk_size[$i]}" "${chunk_index[$i]}" "${epochs[$i]}"
done