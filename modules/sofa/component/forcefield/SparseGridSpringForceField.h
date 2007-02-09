#ifndef SOFA_COMPONENT_FORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_H

#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/topology/MultiResSparseGridTopology.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class SparseGridSpringForceField : public StiffSpringForceField<DataTypes>
{
    /// make the spring force field of a sparsegrid
public:
    typedef StiffSpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Real linesStiffness;
    Real linesDamping;
    Real quadsStiffness;
    Real quadsDamping;
    Real cubesStiffness;
    Real cubesDamping;
    char* filename;
    typedef topology::MultiResSparseGridTopology::SparseGrid Voxels;
    typedef topology::MultiResSparseGridTopology::SparseGrid::Index3D Index3D;


public:
    SparseGridSpringForceField(core::componentmodel::behavior::MechanicalState<DataTypes>* object1, core::componentmodel::behavior::MechanicalState<DataTypes>* object2)
        : StiffSpringForceField<DataTypes>(object1, object2),
          linesStiffness(0), linesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          cubesStiffness(0), cubesDamping(0)
    {
    }

    SparseGridSpringForceField()
        : linesStiffness(0), linesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          cubesStiffness(0), cubesDamping(0)
    {
    }

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    Real getStiffness() const { return linesStiffness; }
    Real getLinesStiffness() const { return linesStiffness; }
    Real getQuadsStiffness() const { return quadsStiffness; }
    Real getCubesStiffness() const { return cubesStiffness; }
    void setStiffness(Real val)
    {
        linesStiffness = val;
        quadsStiffness = val;
        cubesStiffness = val;
    }
    void setLinesStiffness(Real val)
    {
        linesStiffness = val;
    }
    void setQuadsStiffness(Real val)
    {
        quadsStiffness = val;
    }
    void setCubesStiffness(Real val)
    {
        cubesStiffness = val;
    }


    Real getDamping() const { return linesDamping; }
    Real getLinesDamping() const { return linesDamping; }
    Real getQuadsDamping() const { return quadsDamping; }
    Real getCubesDamping() const { return cubesDamping; }
    void setDamping(Real val)
    {
        linesDamping = val;
        quadsDamping = val;
        cubesDamping = val;
    }
    void setLinesDamping(Real val)
    {
        linesDamping = val;
    }
    void setQuadsDamping(Real val)
    {
        quadsDamping = val;
    }
    void setCubesDamping(Real val)
    {
        cubesDamping = val;
    }

    virtual void addForce();

    virtual void addDForce();

    virtual void draw();

    void setFileName(char* name)
    {
        filename = name;
    }

    char * getFileName()const {return filename;}
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
