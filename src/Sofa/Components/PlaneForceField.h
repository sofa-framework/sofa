#ifndef SOFA_COMPONENTS_PLANEFORCEFIELD_H
#define SOFA_COMPONENTS_PLANEFORCEFIELD_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

#include <vector>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class PlaneForceField : public Core::ForceField<DataTypes>, public Abstract::VisualModel
{
public:
    typedef Core::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Deriv planeNormal;
    Real planeD;

    Real stiffness;

    std::vector<unsigned int> contacts;

public:
    PlaneForceField(Core::MechanicalModel<DataTypes>* object=NULL, const std::string& /*name*/="")
        : Core::ForceField<DataTypes>(object), planeD(0), stiffness(500)
    {
    }

    void setPlane(const Deriv& normal, Real d)
    {
        planeNormal = normal;
        planeD = d;
        Real n = normal.norm();
        planeNormal /= n;
        d /= n;
    }

    void setStiffness(Real stiff)
    {
        stiffness = stiff;
    }

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecCoord& x, const VecDeriv& v, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
