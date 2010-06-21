<? include_once("BasicTypes.inc.php");

class TorusFFD extends SOFAObject{
public function TorusFFD(){
		$ab=new AABB(new Coord(-10,-3,-7),new Coord(8,2.5,7));
		$this->setAABB($ab);
}
public function printObj(){
?>
	<Node name="Torus" processor="0">	
                       <Object type="ParallelCGImplicitSolver" iterations="25" threshold="0.000000000001" tolerance="0.000001" rayleighStiffness="0.1" rayleighMass="0" />
		<Object type="MechanicalObject" dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" />
		<Object type="UniformMass" totalmass="10"/>
		<Object type="RegularGrid"
			nx="6" ny="5" nz="2"
			xmin="-7.5" xmax="7.5"
			ymin="-6" ymax="6"
			zmin="-1.75" zmax="1.75"
			/>
		<Object type="RegularGridSpringForceField" name="Springs" stiffness="350" damping="1" />
		<Node name="Visu">
			<Object type="OglModel" name="Visual" fileMesh="mesh/torus2_scale3.obj"   dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" color="blue" />
			<Object type="BarycentricMapping" object1="../.." object2="Visual" />
		</Node>
		<Node name="Surf">
        		<Object type="MeshLoader" name="meshLoader" filename="mesh/torus2_scale3.obj"/>
			<Object type="Mesh" />
			<Object type="MechanicalObject"  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" rz="<?=$this->rz?>" ry="<?=$this->ry?>" />
			<Object type="Triangle" />
			<Object type="BarycentricMapping" />
		</Node>
	</Node>


<?}
};
?>
