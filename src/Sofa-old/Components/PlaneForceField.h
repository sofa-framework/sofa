#ifndef SOFA_COMPONENTS_PLANEFORCEFIELD_H
#define SOFA_COMPONENTS_PLANEFORCEFIELD_H

#include "Sofa-old/Core/ForceField.h"
#include "Sofa-old/Core/MechanicalModel.h"
#include "Sofa-old/Abstract/VisualModel.h"

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
    Real damping;


    std::vector<unsigned int> contacts;

public:


    Common::DataField<Coord> _color;

    PlaneForceField(Core::MechanicalModel<DataTypes>* object=NULL, const std::string& /*name*/="")
        : Core::ForceField<DataTypes>(object), planeD(0), stiffness(500), damping(5)
        , _color(dataField(&_color, Coord(0,.5f,.2f), "color", "plane color"))
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

    void setDamping(Real damp)
    {
        damping = damp;
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

} // namespace Components

} // namespace Sofa

#endif
