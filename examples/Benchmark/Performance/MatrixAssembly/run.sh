#!/bin/bash

# This script executes all the simulation files found in this directory
# It extracts and display simulation times, so they can be compared.

#location of the runSofa executable
SOFA=bin/runSofa

for filename in *.scn; do
  echo $filename

  #run the simulation
  $SOFA -g batch -n 1000 --computationTimeSampling 1000 $filename > "$filename.perf"

  #display the timings
  grep "iterations done in" "$filename.perf"
  grep "LEVEL" "$filename.perf"
  grep "\.\.AnimateVisitor" "$filename.perf"
  stats=$(grep "\.\.AnimateVisitor" "$filename.perf")
  milliseconds="$(echo $stats | cut -d' ' -f6)"
  echo "$milliseconds ms"

  rm "$filename.perf"
done