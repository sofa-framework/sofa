#ifndef SOFA_COMPONENT_FORCEFIELD_LENNARDJONESFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_LENNARDJONESFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class LennardJonesForceField : public sofa::core::componentmodel::behavior::ForceField<DataTypes>, public sofa::core::VisualModel
{
public:
    typedef sofa::core::componentmodel::behavior::ForceField<DataTypes> Inherit;
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
    LennardJonesForceField()
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

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);
    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
