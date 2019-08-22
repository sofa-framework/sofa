/** Deformable liver */

/** Root of the data tree (scene graph)
GNodes define the structure of the scene.
They have basically two lists:
  - components: the modules dedicated to a given task
  - children: GNodes used to model alternative models of the body
Auxiliary lists and singletons are used to rapidly access components by type. Using multiple inheritance, a component may belong to different lists and singletons.
*/
GNode
{
    /** Data available to all components of the GNode */
    dof = MechanicalObject<Particle>; ///< Component holding state vectors: x,v,f,dx,aux_1,...
    topology = MeshTopology;          ///< Component holding edges, triangles, tetrahedra,...
    gravity;                          ///< vec3
    localFrame;                       ///< (translation, rotation)

    /** Independent components */
    mass = UniformMass<Particle,float>; ///< diagonal matrix
    forces = {                          ///< all the forces applied to the particles
        TetrahedronFEMForceField<Particle>; ///< volume forces
        TriangleFEMForceField<Particle>;     ///< surface forces
        UniformMass<Particle,float>;         ///< also in this list by multiple inheritance
    };
    constraints = {                     ///< all the constraints applied to the particles
        FixedConstraint<Particle>;      ///< three fixed points
    };
    odeSolver = EulerSolver;            ///< simple explicit time integrator

    /** Children
    Alternative views of the same body.
    */
    GNode {           ///< Collision model
        /** Data available to all components */
        dof = SphereModel<Particle>;      ///< radii + state vectors: x,v,f,dx,aux_1,...
        topology =;                       ///< no topological data needed here
        gravity;                          ///< = parent
        localFrame;                       ///< = parent

        /** Independent components
        The mechanicalMapping is used to update the the displacements of the child based on the displacements of the parent, and the forces of the parent based on the forces of the child
        */
        mechanicalMapping = MechanicalMapping< MechanicalObject<Particle>, SphereModel<Particle> >;
        mass = ;
        forces = {};
        constraints = {};
        odeSolver = ;
    }
    GNode {           ///< Visual model
        /** Data available to all components */
        dof =;
        topology =;
        gravity;                          ///< = parent
        localFrame;                       ///< = parent

        /** Independent components
        * The mapping is not a MechanicalMapping. It is designed to update the VisualModel based on the parent positions
        */
        visualModel = MyVisualModel;
        Mapping< MechanicalObject<Particle>, MyVisualModel >;
    }
}


