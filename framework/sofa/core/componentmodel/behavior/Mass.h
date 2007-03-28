/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_H

#include <sofa/core/componentmodel/behavior/BaseMass.h>
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
class Mass : public ForceField<DataTypes>, public BaseMass
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    Mass(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Mass();

    virtual void addMDx(); ///< f += M dx

    virtual void accFromF(); ///< dx = M^-1 f

    virtual void addMDx(VecDeriv& f, const VecDeriv& dx) = 0; ///< f += M dx

    virtual void accFromF(VecDeriv& a, const VecDeriv& f) = 0; ///< dx = M^-1 f

    // Mass forces (gravity) often have null derivative
    virtual void addDForce(VecDeriv& /*df*/, const VecDeriv& /*dx*/)
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
This default implementation returns no inertia.
*/
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ )
{
    return Deriv();
    //const Deriv& omega=sv.getAngularVelocity();
    //return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
