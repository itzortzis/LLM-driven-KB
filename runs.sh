#!/bin/bash



chunk_size=(500 500 5000 5000 5000 5000 8000 8000)
chunk_index=(0 5 0 1 2 3 0 1)
epochs=(3 3 3 3 3 3 3 3)

for i in "${!chunk_size[@]}"; do
  echo "Running ${epochs[$i]} epochs with chunk size: ${chunk_size[$i]} and chunk index: ${chunk_index[$i]}..."
  python llm.py "${chunk_size[$i]}" "${chunk_index[$i]}" "${epochs[$i]}"
done