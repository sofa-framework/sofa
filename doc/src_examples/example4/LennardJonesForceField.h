#ifndef LENNARDJONESFORCEFIELD_H
#define LENNARDJONESFORCEFIELD_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

template<class DataTypes>
class LennardJonesForceField : public Sofa::Core::ForceField, public Sofa::Abstract::VisualModel
{
public:
    typedef Sofa::Core::ForceField Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Sofa::Core::MechanicalModel<DataTypes>* object;

    Real a,b,alpha,beta,dmax,fmax;
    Real d0,p0;

    struct DForce
    {
        unsigned int a,b;
        Real df;
    };

    std::vector<DForce> dforces;

public:
    LennardJonesForceField(Sofa::Core::MechanicalModel<DataTypes>* object, const std::string& /*name*/="")
        : object(object), a(1), b(1), alpha(6), beta(12), dmax(2), fmax(1), d0(1), p0(1)
    {
    }

    void setAlpha(Real v) { alpha = v; }
    void setBeta(Real v) { beta = v; }
    void setFMax(Real v) { fmax = v; }
    void setDMax(Real v) { dmax = v; }
    void setD0(Real v) { d0 = v; }
    void setP0(Real v) { p0 = v; }

    virtual void init();

    virtual void addForce();

    virtual void addDForce();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

#endif
