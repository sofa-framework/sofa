<? include_once("BasicTypes.inc.php");

class DragonFFD extends SOFAObject{
public function DragonFFD(){
		$ab=new AABB(new Coord(-11,-7,-4),new Coord(11,7,4));
		$this->setAABB($ab);
}
public function printObj(){
?>
	<Node name="Dragon" processor="<?=$this->processor?>">
                       <Object type="ParallelCGImplicitSolver" iterations="25" threshold="0.000000000001" tolerance="0.000001" rayleighStiffness="0.1" rayleighMass="0" />

		<Object type="MechanicalObject" dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" />
		<Object type="UniformMass" totalmass="10"/>
		<Object type="RegularGrid"
			nx="6" ny="5" nz="3"
			xmin="-11" xmax="11"
			ymin="-7" ymax="7"
			zmin="-4" zmax="4"
			/>
		<Object type="RegularGridSpringForceField" name="Springs" stiffness="350" damping="1" />
		<Node name="Visu">
			<Object type="OglModel" name="Visual" fileMesh="mesh/dragon.obj"  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>"  color="red" />
			<Object type="BarycentricMapping" object1="../.." object2="Visual" />
		</Node>
		<Node name="Surf">
        		<Object type="MeshLoader" name="meshLoader" filename="mesh/dragon.obj"/>
			<Object type="Mesh" />
			<Object type="MechanicalObject"  dx="<?=$this->dx?>" dy="<?=$this->dy?>" dz="<?=$this->dz?>" rx="<?=$this->rx?>" />
			<Object type="Triangle" />
			<Object type="BarycentricMapping" />
		</Node>
	</Node>


<?}
};
?>
