#!/usr/bin/env bash

DATA_DIR=/disk1/medical
LABELS_FILE="${DATA_DIR}/labels.txt"

while read LABEL; do
  if [[ $LABEL == 0 ]]; then
  	echo "0"
  elif [[ $LABEL == 1 ]]; then
  	echo "1"
  fi
done < "${LABELS_FILE}"
