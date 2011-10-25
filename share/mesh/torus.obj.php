<?php
$AXIS=$_ENV["AXIS"]?$_ENV["AXIS"]:"Y";
$R1=$_ENV["R1"]?$_ENV["R1"]:1;
$R2=$_ENV["R2"]?$_ENV["R2"]:3;
$NL=$_ENV["NL"]?$_ENV["NL"]:64;
$NC=$_ENV["NC"]?$_ENV["NC"]:32;

echo "# AXIS=".$AXIS." R1=".$R1." R2=".$R2." NL=".$NL." NC=".$NC." php torus.obj.php\n";

function print_v($x,$y,$z) {
  global $AXIS;
  if ($AXIS=="X")
    echo "v ".$z." ".$x." ".$y."\n";
  else if ($AXIS=="Y")
    echo "v ".$y." ".$z." ".$x."\n";
  else
    echo "v ".$x." ".$y." ".$z."\n";
}

function print_vn($x,$y,$z) {
  global $AXIS;
  if ($AXIS=="X")
    echo "vn ".$z." ".$x." ".$y."\n";
  else if ($AXIS=="Y")
    echo "vn ".$y." ".$z." ".$x."\n";
  else
    echo "vn ".$x." ".$y." ".$z."\n";
}

$RC = ($R1+$R2)/2;
$RT = ($R2-$R1)/2;

for($l=0;$l<$NL;++$l)
{
  $rx = cos(2*M_PI * $l / $NL);
  $ry = sin(2*M_PI * $l / $NL);
  $cx = $rx*$RC;
  $cy = $ry*$RC;
  for($c=0;$c<$NC;++$c)
  {
    $x=$cx + $rx * cos(2*M_PI * $c / $NC) * $RT;
    $y=$cy + $ry * cos(2*M_PI * $c / $NC) * $RT;
    $z=sin(2*M_PI * $c / $NC) * $RT;

    print_v($x,$y,$z);
  }
}

for($l=0;$l<$NL;++$l)
{
  $rx = cos(2*M_PI * $l / $NL);
  $ry = sin(2*M_PI * $l / $NL);
  for($c=0;$c<$NC;++$c)
  {
    $x=$rx * cos(2*M_PI * $c / $NC);
    $y=$ry * cos(2*M_PI * $c / $NC);
    $z=sin(2*M_PI * $c / $NC);

    print_vn($x,$y,$z);
  }
}

for($l=0;$l<$NL;++$l)
{
  $l2 = ($l+1) % $NL;
  for($c=0;$c<$NC;++$c)
  {
    $c2 = ($c+1) % $NC;
    $v1 = 1 + $l  * $NC + $c ;
    $v2 = 1 + $l2 * $NC + $c ;
    $v3 = 1 + $l2 * $NC + $c2;
    $v4 = 1 + $l  * $NC + $c2;

    echo "f ".$v1."//".$v1." ".$v2."//".$v2." ".$v3."//".$v3." ".$v4."//".$v4."\n";
  }
}

?>
