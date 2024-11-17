#!/usr/bin/env bash

calc_params_script="./calculate_z11_run_params.py"
py_out="./new_values.out"
z="$1"
params="$2"

if [ "$#" -lt 2 ]; then
    echo "Error: redshift and parameters file are required." >&2
    echo "Usage: $0 <z> <params> [additional params...]" >&2
    exit 1
fi

change_params_file() {
  params="$1"
  z="$2"
  boxsize_old=$(sed -n 's/^boxsize\s*=\s*\([0-9.eE+-]\+\).*/\1/p' "$params")
  python "$calc_params_script" --z "$z" --boxsize "$boxsize_old" > "$py_out"
  Rsmooth_max=$(grep z="$z" "$py_out" | grep Rsmooth_max | awk -F':' '{print $2}')
  cenz=$(grep z="$z" "$py_out"  | grep cenz | awk -F':' '{print $2}')
  TabInterpX2=$(grep z="$z" "$py_out"  | grep TabInterpX2 | awk -F':' '{print $2}')
  boxsize=$(grep z="$z" "$py_out"  | grep boxsize | awk -F':' '{print $2}')
  #echo "$Rsmooth_max"
  #echo "$cenz"
  #echo "$TabInterpX2"
  #echo "$boxsize"
  new_params=$(echo "$params" | sed "s/\.ini/_z${z}.ini/")
  cp "$params" "$new_params"
  sed -i "s/^\(boxsize[[:space:]]*\)=.*/\1= ${boxsize}/" "$new_params"
  sed -i "s/^\(cenz[[:space:]]*\)=.*/\1= ${cenz}/" "$new_params"
  sed -i "s/^\(TabInterpX2[[:space:]]*\)=.*/\1= ${TabInterpX2}/" "$new_params"
  sed -i "s/^\(Rsmooth_max[[:space:]]*\)=.*/\1= ${Rsmooth_max}/" "$new_params"
}

change_params_file "$params" "$z"
