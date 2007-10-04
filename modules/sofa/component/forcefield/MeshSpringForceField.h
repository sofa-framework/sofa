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
    DataField< Real >  linesStiffness;
    DataField< Real >  linesDamping;
    DataField< Real >  trianglesStiffness;
    DataField< Real >  trianglesDamping;
    DataField< Real >  quadsStiffness;
    DataField< Real >  quadsDamping;
    DataField< Real >  tetrasStiffness;
    DataField< Real >  tetrasDamping;
    DataField< Real >  cubesStiffness;
    DataField< Real >  cubesDamping;

    void addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping);

public:
    MeshSpringForceField()
        : linesStiffness(dataField(&linesStiffness,Real(0),"linesStiffness","Stiffness for the Lines")),
          linesDamping(dataField(&linesDamping,Real(0),"linesDamping","Damping for the Lines")),
          trianglesStiffness(dataField(&trianglesStiffness,Real(0),"trianglesStiffness","Stiffness for the Triangles")),
          trianglesDamping(dataField(&trianglesDamping,Real(0),"trianglesDamping","Damping for the Triangles")),
          quadsStiffness(dataField(&quadsStiffness,Real(0),"quadsStiffness","Stiffness for the Quads")),
          quadsDamping(dataField(&quadsDamping,Real(0),"quadsDamping","Damping for the Quads")),
          tetrasStiffness(dataField(&tetrasStiffness,Real(0),"tetrasStiffness","Stiffness for the Tetras")),
          tetrasDamping(dataField(&tetrasDamping,Real(0),"tetrasDamping","Damping for the Tetras")),
          cubesStiffness(dataField(&cubesStiffness,Real(0),"cubesStiffness","Stiffness for the Cubes")),
          cubesDamping(dataField(&cubesDamping,Real(0),"cubesDamping","Damping for the Cubes"))
    {
    }

    virtual double getPotentialEnergy();


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
