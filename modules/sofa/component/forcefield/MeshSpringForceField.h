#ifndef SOFA_COMPONENT_FORCEFIELD_MESHSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_MESHSPRINGFORCEFIELD_H

#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <set>

namespace sofa
{

namespace component
{

namespace forcefield
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
    Data< Real >  linesStiffness;
    Data< Real >  linesDamping;
    Data< Real >  trianglesStiffness;
    Data< Real >  trianglesDamping;
    Data< Real >  quadsStiffness;
    Data< Real >  quadsDamping;
    Data< Real >  tetrasStiffness;
    Data< Real >  tetrasDamping;
    Data< Real >  cubesStiffness;
    Data< Real >  cubesDamping;

    void addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping);

public:
    MeshSpringForceField()
        : linesStiffness(initData(&linesStiffness,Real(0),"linesStiffness","Stiffness for the Lines")),
          linesDamping(initData(&linesDamping,Real(0),"linesDamping","Damping for the Lines")),
          trianglesStiffness(initData(&trianglesStiffness,Real(0),"trianglesStiffness","Stiffness for the Triangles")),
          trianglesDamping(initData(&trianglesDamping,Real(0),"trianglesDamping","Damping for the Triangles")),
          quadsStiffness(initData(&quadsStiffness,Real(0),"quadsStiffness","Stiffness for the Quads")),
          quadsDamping(initData(&quadsDamping,Real(0),"quadsDamping","Damping for the Quads")),
          tetrasStiffness(initData(&tetrasStiffness,Real(0),"tetrasStiffness","Stiffness for the Tetras")),
          tetrasDamping(initData(&tetrasDamping,Real(0),"tetrasDamping","Damping for the Tetras")),
          cubesStiffness(initData(&cubesStiffness,Real(0),"cubesStiffness","Stiffness for the Cubes")),
          cubesDamping(initData(&cubesDamping,Real(0),"cubesDamping","Damping for the Cubes"))
    {
    }

    virtual sofa::defaulttype::Vector3::value_type getPotentialEnergy();


    Real getStiffness() const { return linesStiffness.getValue(); }
    Real getLinesStiffness() const { return linesStiffness.getValue(); }
    Real getTrianglesStiffness() const { return trianglesStiffness.getValue(); }
    Real getQuadsStiffness() const { return quadsStiffness.getValue(); }
    Real getTetrasStiffness() const { return tetrasStiffness.getValue(); }
    Real getCubesStiffness() const { return cubesStiffness.getValue(); }
    void setStiffness(Real val)
    {
        linesStiffness.setValue(val);
        trianglesStiffness.setValue(val);
        quadsStiffness.setValue(val);
        tetrasStiffness.setValue(val);
        cubesStiffness.setValue(val);
    }
    void setLinesStiffness(Real val)
    {
        linesStiffness.setValue(val);
    }
    void setTrianglesStiffness(Real val)
    {
        trianglesStiffness.setValue(val);
    }
    void setQuadsStiffness(Real val)
    {
        quadsStiffness.setValue(val);
    }
    void setTetrasStiffness(Real val)
    {
        tetrasStiffness.setValue(val);
    }
    void setCubesStiffness(Real val)
    {
        cubesStiffness.setValue(val);
    }

    Real getDamping() const { return linesDamping.getValue(); }
    Real getLinesDamping() const { return linesDamping.getValue(); }
    Real getTrianglesDamping() const { return trianglesDamping.getValue(); }
    Real getQuadsDamping() const { return quadsDamping.getValue(); }
    Real getTetrasDamping() const { return tetrasDamping.getValue(); }
    Real getCubesDamping() const { return cubesDamping.getValue(); }
    void setDamping(Real val)
    {
        linesDamping.setValue(val);
        trianglesDamping.setValue(val);
        quadsDamping.setValue(val);
        tetrasDamping.setValue(val);
        cubesDamping.setValue(val);
    }
    void setLinesDamping(Real val)
    {
        linesDamping.setValue(val);
    }
    void setTrianglesDamping(Real val)
    {
        trianglesDamping.setValue(val);
    }
    void setQuadsDamping(Real val)
    {
        quadsDamping.setValue(val);
    }
    void setTetrasDamping(Real val)
    {
        tetrasDamping.setValue(val);
    }
    void setCubesDamping(Real val)
    {
        cubesDamping.setValue(val);
    }

    virtual void init();

    void parse(core::objectmodel::BaseObjectDescription* arg);

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
