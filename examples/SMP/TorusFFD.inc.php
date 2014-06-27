<? include_once("BasicTypes.inc.php");

class TorusFFD extends SOFAObject{
public function TorusFFD(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>
	<Node name="Torus" processor="0">	
    <EulerImplicitSolver rayleighStiffness="0.1" rayleighMass="0" />
    <ParallelCGLinearSolver iterations="25" threshold="0.000000000001" tolerance="0.000001" />
		<MechanicalObject dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" />
		<UniformMass totalmass="10"/>
		<RegularGrid
			nx="6" ny="5" nz="2"
			xmin="-7.5" xmax="7.5"
			ymin="-6" ymax="6"
			zmin="-1.75" zmax="1.75"
			/>
		<RegularGridSpringForceField name="Springs" stiffness="350" damping="1" />
		<Node name="Visu">
			<OglModel name="Visual" fileMesh="mesh/torus2_scale3.obj"   dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" color="blue" />
			<BarycentricMapping input="@.." output="@Visual" />
		</Node>
		<Node name="Surf">
     	<MeshLoader name="meshLoader" filename="mesh/torus2_scale3.obj"/>
			<Mesh />
			<MechanicalObject  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" />
			<Triangle />
			<BarycentricMapping />
		</Node>
	</Node>


<?}
};
?>
