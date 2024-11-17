#!/usr/bin/env bash

calc_params_script="./calculate_different_z_params.py"
py_out="./new_values.out"
z="$1"
params="$2"

if [ "$#" -lt 2 ]; then
    echo "This is a code to change the parameters of a parameters.ini file" >&1
	echo "to run the same sim at different redshift." >&1
	echo "New files will be created with filename parameters_z11.ini," >&1
	echo "for z=11, for instance." >&1
	echo "" >&1
    echo "Error: redshift and parameters file are required." >&2
    echo "Usage: $0 <z> <params> [additional params...]" >&2
    exit 1
fi

change_params_file() {
  z="$1"
  params="$2"
echo "Successfully enter"
  boxsize_old=$(sed -n 's/^boxsize\s*=\s*\([0-9.eE+-]\+\).*/\1/p' "$params")
  nmesh=$(sed -n 's/^nmesh\s*=\s*\([0-9.eE+-]\+\).*/\1/p' "$params")
  ntile=$(sed -n 's/^ntile\s*=\s*\([0-9.eE+-]\+\).*/\1/p' "$params")
  nbuff=$(sed -n 's/^nbuff\s*=\s*\([0-9.eE+-]\+\).*/\1/p' "$params")
  python "$calc_params_script" --z "$z" --boxsize "$boxsize_old" > "$py_out" 
  Rsmooth_max=$(grep z="$z" "$py_out" | grep Rsmooth_max | awk -F':' '{print $2}')
  cenz=$(grep z="$z" "$py_out"  | grep cenz | awk -F':' '{print $2}')
  TabInterpX2=$(grep z="$z" "$py_out"  | grep TabInterpX2 | awk -F':' '{print $2}')
  boxsize=$(grep z="$z" "$py_out"  | grep boxsize | awk -F':' '{print $2}')
  run_name="n${nmesh}_nb${nbuff}_nt${ntile}_z${z}"
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
  sed -i "s/^\(run_name[[:space:]]*\)=.*/\1= ${run_name}/" "$new_params"
}

echo "Successfully before"
change_params_file "$z" "$params" 
