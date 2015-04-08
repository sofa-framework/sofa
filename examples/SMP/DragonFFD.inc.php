<? include_once("BasicTypes.inc.php");

class DragonFFD extends SOFAObject{
public function DragonFFD(){
		$ab=new AABB(new Coord(-11,-7,-4),new Coord(11,7,4));
		$this->setAABB($ab);
}
public function printObj(){
?>
	<Node name="Dragon" processor="<?=$this->processor?>">
    <EulerImplicitSolver rayleighStiffness="0.1" rayleighMass="0" />
    <ParallelCGLinearSolver iterations="25" threshold="0.000000000001" tolerance="0.000001" />

		<MechanicalObject dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" />
		<UniformMass totalmass="10"/>
		<RegularGrid
			nx="6" ny="5" nz="3"
			xmin="-11" xmax="11"
			ymin="-7" ymax="7"
			zmin="-4" zmax="4"
			/>
		<RegularGridSpringForceField name="Springs" stiffness="350" damping="1" />
		<Node name="Visu">
			<OglModel name="Visual" fileMesh="mesh/dragon.obj"  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>"  color="red" />
			<BarycentricMapping input="@.." output="@Visual" />
		</Node>
		<Node name="Surf">
   		<MeshLoader name="meshLoader" filename="mesh/dragon.obj"/>
			<Mesh />
			<MechanicalObject  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" />
			<Triangle />
			<BarycentricMapping />
		</Node>
	</Node>


<?}
};
?>
