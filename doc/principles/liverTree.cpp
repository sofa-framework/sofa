/** mechanical system (liver) based on 3D points */
GNode
{
    /** Data available to all components */
    dof = MechanicalObject<Particle>; ///< Component holding state vectors: x,v,f,dx,aux_1,...
    topology = MeshTopology;          ///< Component holding edges, triangles, tetrahedra,...
    gravity;                          ///< vec3
    localFrame;                       ///< (translation, rotation)

    /** Independent components */
    mass = UniformMass<Particle,float>; ///< diagonal matrix
    forces = {                          ///< all the forces applied to the particles
        TetrahedronFEMForceField<Particle>; ///< volume forces
        TriangleFEMForceField<Particle>;     ///< surface forces
    };
    constraints = {                     ///< all the constraints applied to the particles
        FixedConstraint<Particle>;      ///< three fixed points
    };
    odeSolver = EulerSolver;            ///< simple explicit time integrator
}
