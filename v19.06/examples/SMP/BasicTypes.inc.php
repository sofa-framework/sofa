<?php
class Coord{
var	$x;
var	$y;
var	$z;
	public function Coord($_x,$_y,$_z){
	$this->x=$_x;
	$this->y=$_y;
	$this->z=$_z;
	}
};
class AABB{
	var $min;
	var $max;
	public function AABB($_min,$_max){
	$this->min=$_min;
	$this->max=$_max;
	}
};
class SOFAObject{
	var $dx=0;
	var $dy=0;
	var $dz=0;
	var $rx=0;
	var $ry=0;
	var $rz=0;
	var $processor=1;
	var $aabb;
  var $index;
  var $scale;

public function setAABB($_ab){
$this->aabb=$_ab;
}
public function setDx($_dx){

	$this->dx=$_dx;
	$this->aabb->min->x+=$_dx;
	$this->aabb->max->x+=$_dx;

}
public function setDy($_dy){

	$this->dy=$_dy;
	$this->aabb->min->y+=$_dy;
	$this->aabb->max->y+=$_dy;

}
public function setDz($_dz){

	$this->dz=$_dz;
	$this->aabb->min->z+=$_dz;
	$this->aabb->max->z+=$_dz;

}
public function setRx($_dx){

	$this->rx=$_dx;

}
public function setRy($_dy){

	$this->ry=$_dy;

}
public function setRz($_dz){

	$this->rz=$_dz;

}
};
?>

