#!/bin/bash

(
echo '<?xml version="1.0" ?>'
echo '<Node>'
let d=1
for i in $*;
do
let d-=1
done
for i in $*;
do
echo '  <Node name="'$i'" >'
echo '    <OglModel filename="'$i'" dx="'$d'" />'
echo '  </Node>'
let d+=2
done
echo '</Node>'
) > tmp.scn
runSofa tmp.scn
