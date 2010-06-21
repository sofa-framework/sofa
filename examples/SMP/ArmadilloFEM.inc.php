<? include_once("BasicTypes.inc.php");

class ArmadilloFEM extends SOFAObject{
public function ArmadilloFEM(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>

	 <include name="ArmadilloFEM"         href="Objects/SMP/GridFEMSphereCPU.xml"  n="8 8 8"      dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" sparse__filename="mesh/Armadillo_verysimplified.msh" VisualMesh__filename="mesh/Armadillo_verysimplified.obj" collisionLoader__filename="mesh/Armadillo_verysimplified.obj"  />
<?}
};
?>
