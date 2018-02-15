/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_BaseMaterial_H
#define FLEXIBLE_BaseMaterial_H

#include <sofa/defaulttype/Mat.h>
#include "../quadrature/BaseGaussPointSampler.h"

// Hack to fix compilation of mixing types to remove once the macro for constants is setted up
inline float pow(float f, double exp)
{
	return powf(f, static_cast<double>(exp));
}

namespace sofa
{
namespace defaulttype
{

/** Template class used to implement one Material block
*/
template<class _T>
class BaseMaterialBlock
{
public:
    typedef _T T;
    typedef typename T::Coord Coord;
    typedef typename T::Deriv Deriv;
    typedef typename T::Real Real;

    typedef component::engine::BaseGaussPointSampler::volumeIntegralType volumeIntegralType;
    typedef Mat<T::deriv_total_size,T::deriv_total_size,Real> MatBlock;  ///< stiffness or compliance matrix block

    // quadrature data (from a GaussPointSampler)
    const volumeIntegralType* volume;

    // compute U(x)
    virtual Real getPotentialEnergy(const Coord& x) const = 0;
    // compute $ f=-dU/dx + f(v) $
    virtual void addForce( Deriv& f , const Coord& x , const Deriv& v) const = 0;
    // compute $ df += kFactor K dx + bFactor B dx $
    virtual void addDForce( Deriv&   df , const Deriv&   dx, const SReal& kfactor, const SReal& bfactor ) const = 0;

    virtual MatBlock getK() const = 0;
    virtual MatBlock getB() const = 0;
    virtual MatBlock getC() const = 0;
};




} // namespace defaulttype
} // namespace sofa



#endif
