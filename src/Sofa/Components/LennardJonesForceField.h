#ifndef SOFA_COMPONENTS_LENNARDJONESFORCEFIELD_H
#define SOFA_COMPONENTS_LENNARDJONESFORCEFIELD_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

#include <vector>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class LennardJonesForceField : public Sofa::Core::ForceField<DataTypes>, public Sofa::Abstract::VisualModel
{
public:
    typedef Sofa::Core::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Real a,b,alpha,beta,dmax,fmax;
    Real d0,p0;
    Real damping;

    struct DForce
    {
        unsigned int a,b;
        Real df;
    };

    std::vector<DForce> dforces;

public:
    LennardJonesForceField(Sofa::Core::MechanicalModel<DataTypes>* /*object*/=NULL)
        : a(1), b(1), alpha(6), beta(12), dmax(2), fmax(1), d0(1), p0(1), damping(0)
    {
    }

    void setAlpha(Real v) { alpha = v; }
    void setBeta(Real v) { beta = v; }
    void setFMax(Real v) { fmax = v; }
    void setDMax(Real v) { dmax = v; }
    void setD0(Real v) { d0 = v; }
    void setP0(Real v) { p0 = v; }
    void setDamping(Real v) { damping = v; }

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

};

} // namespace Sofa

} // namespace Components

#endif
