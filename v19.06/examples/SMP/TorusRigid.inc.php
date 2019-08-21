<? include_once("BasicTypes.inc.php");

class TorusRigid  extends SOFAObject{
public function TorusRigid(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>

	 <include name="Rigid"         href="Objects/SMP/TorusRigid.xml"        dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>"  scale="3.0"    />

<?}
};
?>
