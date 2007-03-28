#ifndef SOFA_COMPONENT_FORCEFIELD_SPHEREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPHEREFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/DataField.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SphereForceFieldInternalData
{
public:
};

template<class DataTypes>
class SphereForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    struct Contact
    {
        int index;
        Coord normal;
        Real fact;
    };

    std::vector< Contact > contacts;

    SphereForceFieldInternalData<DataTypes> data;

public:

    DataField<Coord> sphereCenter;
    DataField<Real> sphereRadius;
    DataField<Real> stiffness;
    DataField<Real> damping;
    DataField<defaulttype::Vec3f> color;
    DataField<bool> bDraw;

    SphereForceField()
        : sphereCenter(dataField(&sphereCenter, "center", "sphere center"))
        , sphereRadius(dataField(&sphereRadius, (Real)1, "radius", "sphere radius"))
        , stiffness(dataField(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(dataField(&damping, (Real)5, "damping", "force damping"))
        , color(dataField(&color, defaulttype::Vec3f(0.0f,.5f,.2f), "color", "plane color"))
        , bDraw(dataField(&bDraw, true, "draw", "enable/disable drawing of plane"))
    {
    }

    void setSphere(const Coord& center, Real radius)
    {
        sphereCenter.setValue( center );
        sphereRadius.setValue( radius );
    }

    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    virtual void updateStiffness( const VecCoord& x );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
