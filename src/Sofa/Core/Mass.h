#ifndef SOFA_CORE_MASS_H
#define SOFA_CORE_MASS_H

#include "BasicMass.h"
#include "ForceField.h"
#include "MechanicalModel.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
class Mass : public ForceField<DataTypes>, public BasicMass
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    Mass(MechanicalModel<DataTypes> *mm = NULL);

    virtual ~Mass();

    virtual void addMDx(); ///< f += M dx

    virtual void accFromF(); ///< dx = M^-1 f

    virtual void addMDx(VecDeriv& f, const VecDeriv& dx) = 0; ///< f += M dx

    virtual void accFromF(VecDeriv& a, const VecDeriv& f) = 0; ///< dx = M^-1 f

    // Mass forces (gravity) often have null derivative
    virtual void addDForce(VecDeriv& /*df*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const VecDeriv& /*dx*/)
    {}

    virtual double getKineticEnergy();  ///< vMv/2 using dof->getV()
    virtual double getKineticEnergy( const VecDeriv& v )=0;  ///< vMv/2 using dof->getV()
};

/** Return the inertia force applied to a body referenced in a moving coordinate system.
\param sv spatial velocity (omega, vorigin) of the coordinate system
\param a acceleration of the origin of the coordinate system
\param m mass of the body
\param x position of the body in the moving coordinate system
\param v velocity of the body in the moving coordinate system
This default implementation is for particles. Rigid bodies will be handled in a template specialization.
*/
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& sv, const Vec& a, const M& m, const Coord& x, const Deriv& v )
{
    const Deriv& omega=sv.getAngularVelocity();
    //std::cerr<<"inertiaForce, sv = "<<sv<<", omega ="<<omega<<", a = "<<a<<", m= "<<m<<", x= "<<x<<", v= "<<v<<std::endl;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

} // namespace Core

} // namespace Sofa

#endif
