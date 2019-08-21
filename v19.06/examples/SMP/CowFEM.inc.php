<? include_once("BasicTypes.inc.php");

class CowFEM extends SOFAObject{
public function CowFEM(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>
  <include name="CowFEM"         href="Objects/SMP/GridFEMSphereCPU.xml" scale="12"       dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" collisionLoader__filename="mesh/cow-900.obj" sparse__filename="mesh/cow-900.obj" VisualMesh__filename="mesh/cow-1500.obj"   />



<?}
};
?>
