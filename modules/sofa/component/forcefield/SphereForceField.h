#ifndef SOFA_COMPONENT_FORCEFIELD_SPHEREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPHEREFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>

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
class SphereForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    class Contact
    {
    public:
        int index;
        Coord normal;
        Real fact;
        Contact( int index=0, Coord normal=Coord(),Real fact=Real(0))
            : index(index),normal(normal),fact(fact)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Contact& c )
        {
            in>>c.index>>c.normal>>c.fact;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c )
        {
            out << c.index << " " << c.normal << " " << c.fact ;
            return out;
        }

    };

    Data<sofa::helper::vector<Contact> > contacts;

    SphereForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> sphereCenter;
    Data<Real> sphereRadius;
    Data<Real> stiffness;
    Data<Real> damping;
    Data<defaulttype::Vec3f> color;
    Data<bool> bDraw;

    SphereForceField()
        : contacts(initData(&contacts,"contacts", "Contacts"))
        , sphereCenter(initData(&sphereCenter, "center", "sphere center"))
        , sphereRadius(initData(&sphereRadius, (Real)1, "radius", "sphere radius"))
        , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(initData(&damping, (Real)5, "damping", "force damping"))
        , color(initData(&color, defaulttype::Vec3f(0.0f,0.0f,1.0f), "color", "sphere color"))
        , bDraw(initData(&bDraw, true, "draw", "enable/disable drawing of the sphere"))
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

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

    virtual double getPotentialEnergy(const VecCoord& x);

    virtual void updateStiffness( const VecCoord& x );

    void draw();
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
