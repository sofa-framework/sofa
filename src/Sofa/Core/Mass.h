#ifndef SOFA_CORE_MASS_H
#define SOFA_CORE_MASS_H

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class Mass : public virtual Abstract::BaseObject
{
public:
    virtual ~Mass() { }

    virtual void addMDx()=0; ///< f += M dx

    virtual void accFromF()=0; ///< dx = M^-1 f

    virtual void computeForce() { }

    virtual void computeDf() { }
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
    const Deriv& omega=sv.lineVec;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}


} // namespace Core

} // namespace Sofa

#endif
