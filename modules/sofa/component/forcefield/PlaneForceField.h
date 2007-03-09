#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H

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
class PlaneForceFieldInternalData
{
public:
};

template<class DataTypes>
class PlaneForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    std::vector<unsigned int> contacts;

    PlaneForceFieldInternalData<DataTypes> data;

public:

    DataField<Deriv> planeNormal;
    DataField<Real> planeD;
    DataField<Real> stiffness;
    DataField<Real> damping;
    DataField<Coord> color;
    DataField<bool> bDraw;

    PlaneForceField()
        : planeNormal(dataField(&planeNormal, Deriv(0, 0, 1), "normal", "plane normal"))
        , planeD(dataField(&planeD, (Real)0, "d", "plane d coef"))
        , stiffness(dataField(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(dataField(&damping, (Real)5, "damping", "force damping"))
        , color(dataField(&color, Coord(0.0f,.5f,.2f), "color", "plane color"))
        , bDraw(dataField(&bDraw, false, "draw", "enable/disable drawing of plane"))
    {
    }

    void setPlane(const Deriv& normal, Real d)
    {
        Real n = normal.norm();
        planeNormal.setValue( normal / n);
        planeD.setValue( d / n );
    }

    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    void rotate( Deriv axe, Real angle ); // around the origin (0,0,0)

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    virtual void updateStiffness( const VecCoord& x );

    // -- VisualModel interface
    void draw();
    void draw2(float size=1000.0f);
    void initTextures() { }
    void update() { }
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
