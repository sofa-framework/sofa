#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include "Sofa/Contrib/Testing/FEMcontact/TetrahedralFEMForceField.h"

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
class PlaneForceField : public Core::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef Core::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Core::MechanicalState<DataTypes>* object;
    TetrahedralFEMForceField<DataTypes>* fem;

    Deriv planeNormal;
    Real planeD;

    Real stiffness;

    std::vector<unsigned int> contacts;

    VecDeriv _force;

public:
    PlaneForceField(Core::MechanicalState<DataTypes>* object, const std::string& /*name*/="")
        : object(object), planeD(0), stiffness(500), fem(0)
    {
    }

    PlaneForceField(Core::MechanicalState<DataTypes>* object, TetrahedralFEMForceField<DataTypes>* fem)
        : object(object), fem(fem), planeD(0), stiffness(500)
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

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
