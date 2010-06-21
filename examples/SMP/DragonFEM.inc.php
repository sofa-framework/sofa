<? include_once("BasicTypes.inc.php");

class DragonFEM extends SOFAObject{
public function DragonFEM(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>

	 <include name="DragonFEM"         href="Objects/SMP/GridFEMSphereCPU.xml"        dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>"  filename="mesh/dragon.obj" scale="0.6"  />

<?}
};
?>
