#ifndef SOFA_COMPONENTS_GRIDSPRINGFORCEFIELD_H
#define SOFA_COMPONENTS_GRIDSPRINGFORCEFIELD_H

#include "Sofa/Components/StiffSpringForceField.h"
#include "TrimmedRegularGridTopology.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class RegularGridSpringForceField : public StiffSpringForceField<DataTypes>
{
    double m_potentialEnergy;
public:
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

public:
    RegularGridSpringForceField(Core::MechanicalModel<DataTypes>* object1, Core::MechanicalModel<DataTypes>* object2)
        : StiffSpringForceField<DataTypes>(object1, object2),
          linesStiffness(0), linesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          cubesStiffness(0), cubesDamping(0),
          topology(NULL), trimmedTopology(NULL)
    {
    }

    RegularGridSpringForceField(Core::MechanicalModel<DataTypes>* object)
        : StiffSpringForceField<DataTypes>(object),
          linesStiffness(0), linesDamping(0),
          quadsStiffness(0), quadsDamping(0),
          cubesStiffness(0), cubesDamping(0),
          topology(NULL), trimmedTopology(NULL)
    {
    }

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

    virtual void init();

    virtual void addForce();

    virtual void addDForce();

    virtual double getPotentialEnergy() { return m_potentialEnergy; }

    virtual void draw();

protected:
    RegularGridTopology* topology;
    TrimmedRegularGridTopology* trimmedTopology;
};

} // namespace Components

} // namespace Sofa

#endif
