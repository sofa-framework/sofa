#ifndef SOFA_COMPONENTS_MESHSPRINGFORCEFIELD_H
#define SOFA_COMPONENTS_MESHSPRINGFORCEFIELD_H

#include "Sofa-old/Components/StiffSpringForceField.h"
#include <set>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class MeshSpringForceField : public StiffSpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Real linesStiffness;
    Real linesDamping;
    Real trianglesStiffness;
    Real trianglesDamping;
    Real quadsStiffness;
    Real quadsDamping;
    Real tetrasStiffness;
    Real tetrasDamping;
    Real cubesStiffness;
    Real cubesDamping;

    void addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping);

public:
    MeshSpringForceField(Core::MechanicalModel<DataTypes>* object1, Core::MechanicalModel<DataTypes>* object2)
        : StiffSpringForceField<DataTypes>(object1, object2),
          linesStiffness(0), linesDamping(0),
          trianglesStiffness(0),  trianglesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          tetrasStiffness(0), tetrasDamping(0),
          cubesStiffness(0), cubesDamping(0)
    {
    }

    MeshSpringForceField(Core::MechanicalModel<DataTypes>* object)
        : StiffSpringForceField<DataTypes>(object),
          linesStiffness(0), linesDamping(0),
          trianglesStiffness(0),  trianglesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          tetrasStiffness(0), tetrasDamping(0),
          cubesStiffness(0), cubesDamping(0)
    {
    }

    virtual double getPotentialEnergy();


    Real getStiffness() const { return linesStiffness; }
    Real getLinesStiffness() const { return linesStiffness; }
    Real getTrianglesStiffness() const { return trianglesStiffness; }
    Real getQuadsStiffness() const { return quadsStiffness; }
    Real getTetrasStiffness() const { return tetrasStiffness; }
    Real getCubesStiffness() const { return cubesStiffness; }
    void setStiffness(Real val)
    {
        linesStiffness = val;
        trianglesStiffness = val;
        quadsStiffness = val;
        tetrasStiffness = val;
        cubesStiffness = val;
    }
    void setLinesStiffness(Real val)
    {
        linesStiffness = val;
    }
    void setTrianglesStiffness(Real val)
    {
        trianglesStiffness = val;
    }
    void setQuadsStiffness(Real val)
    {
        quadsStiffness = val;
    }
    void setTetrasStiffness(Real val)
    {
        tetrasStiffness = val;
    }
    void setCubesStiffness(Real val)
    {
        cubesStiffness = val;
    }

    Real getDamping() const { return linesDamping; }
    Real getLinesDamping() const { return linesDamping; }
    Real getTrianglesDamping() const { return trianglesDamping; }
    Real getQuadsDamping() const { return quadsDamping; }
    Real getTetrasDamping() const { return tetrasDamping; }
    Real getCubesDamping() const { return cubesDamping; }
    void setDamping(Real val)
    {
        linesDamping = val;
        trianglesDamping = val;
        quadsDamping = val;
        tetrasDamping = val;
        cubesDamping = val;
    }
    void setLinesDamping(Real val)
    {
        linesDamping = val;
    }
    void setTrianglesDamping(Real val)
    {
        trianglesDamping = val;
    }
    void setQuadsDamping(Real val)
    {
        quadsDamping = val;
    }
    void setTetrasDamping(Real val)
    {
        tetrasDamping = val;
    }
    void setCubesDamping(Real val)
    {
        cubesDamping = val;
    }

    virtual void init();
};

} // namespace Components

} // namespace Sofa

#endif
