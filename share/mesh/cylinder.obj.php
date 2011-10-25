<?php
$AXIS=$_ENV["AXIS"]?$_ENV["AXIS"]:"Z";
$R=$_ENV["R"]?$_ENV["R"]:1;
$L=$_ENV["L"]?$_ENV["L"]:10;
$NR=$_ENV["NR"]?$_ENV["NR"]:2;
$NL=$_ENV["NL"]?$_ENV["NL"]:10;
$NC=$_ENV["NC"]?$_ENV["NC"]:32;

echo "# AXIS=".$AXIS." R=".$R." L=".$L." NR=".$NR." NL=".$NL." NC=".$NC." php cylinder.obj.php\n";

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

$z=0.0;
$x=0.0;
$y=0.0;

print_v($x,$y,$z);
for($r=1;$r<=$NR;++$r)
{
  for($c=0;$c<$NC;++$c)
  {
    $x = cos(2*M_PI * $c / $NC) * ($r*$R)/$NR;
    $y = sin(2*M_PI * $c / $NC) * ($r*$R)/$NR;
    print_v($x,$y,$z);
  }
}
for($l=1;$l<$NL;++$l)
{
  $z = $l * $L / $NL;
  for($c=0;$c<$NC;++$c)
  {
    $x = cos(2*M_PI * $c / $NC) * $R;
    $y = sin(2*M_PI * $c / $NC) * $R;
    print_v($x,$y,$z);
  }
}
$z=$L;
for($r=$NR;$r>=1;--$r)
{
  for($c=0;$c<$NC;++$c)
  {
    $x = cos(2*M_PI * $c / $NC) * ($r*$R)/$NR;
    $y = sin(2*M_PI * $c / $NC) * ($r*$R)/$NR;
    print_v($x,$y,$z);
  }
}
$x=0.0;
$y=0.0;
print_v($x,$y,$z);

print_vn(0,0,-1);
  for($c=0;$c<$NC;++$c)
  {
    $x = cos(2*M_PI * $c / $NC);
    $y = sin(2*M_PI * $c / $NC);
    $z = 0;
    print_vn($x,$y,$z);
  }
print_vn(0,0,1);

$n = 1;
for($r=1;$r<=$NR;++$r)
{
  $v0 = 2+($r-1)*$NC;
  for($c=0;$c<$NC;++$c)
  {
      $c2 = ($c+1)%$NC;
    if ($r == 1)
      echo "f 1//".$n." ".($v0+$c2)."//".$n." ".($v0+$c)."//".$n."\n";
    else
      echo "f ".($v0+$c)."//".$n." ".($v0-$NC+$c)."//".$n." ".($v0-$NC+$c2)."//".$n." ".($v0+$c2)."//".$n."\n";
  }
}

for($l=0;$l<$NL;++$l)
{
  $v0 = 2 + ($NR-1)*$NC + $l*$NC;
  for($c=0;$c<$NC;++$c)
  {
      $c2 = ($c+1)%$NC;
      $n = 2 + $c;
      $n2 = 2 + $c2;
      
      echo "f ".($v0+$NC+$c)."//".$n." ".($v0+$c)."//".$n." ".($v0+$c2)."//".$n2." ".($v0+$NC+$c2)."//".$n2."\n";
  }
}

$n = 2 + $NC;
$v2 = 2 + $NC + 2*($NR-1)*$NC + $NL*$NC;
for($r=$NR;$r>=1;--$r)
{
  $v0 = $v2-$r*$NC;
  for($c=0;$c<$NC;++$c)
  {
      $c2 = ($c+1)%$NC;
    if ($r == 1)
      echo "f ".$v2."//".$n." ".($v0+$c)."//".$n." ".($v0+$c2)."//".$n."\n";
    else
      echo "f ".($v0+$NC+$c)."//".$n." ".($v0+$c)."//".$n." ".($v0+$c2)."//".$n." ".($v0+$NC+$c2)."//".$n."\n";
  }
}

?>
