#ifndef SOFA_COMPONENTS_PLANEFORCEFIELD_H
#define SOFA_COMPONENTS_PLANEFORCEFIELD_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class PlaneForceField : public Core::ForceField, public Abstract::VisualModel
{
public:
    typedef Core::ForceField Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Core::MechanicalModel<DataTypes>* object;

    Deriv planeNormal;
    Real planeD;

    Real stiffness;

    std::vector<unsigned int> contacts;

public:
    PlaneForceField(Core::MechanicalModel<DataTypes>* object, const std::string& /*name*/="")
        : object(object), planeD(0), stiffness(500)
    {
    }

    void setPlane(const Deriv& normal, Real d)
    {
        planeNormal = normal;
        planeD = d;
    }

    void setStiffness(Real stiff)
    {
        stiffness = stiff;
    }

    virtual void addForce();

    virtual void addDForce();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
