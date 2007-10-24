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

/**
 *  \brief Component responsible for mass-related computations (gravity, acceleration).
 *
 *  Mass can be defined either as a scalar, vector, or a full mass-matrix.
 *  It is responsible for converting forces to accelerations (for explicit integrators),
 *  or displacements to forces (for implicit integrators).
 *
 *  It is also a ForceField, computing gravity-related forces.
 */
template<class DataTypes>
class Mass : virtual public ForceField<DataTypes>, public BaseMass
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    Mass(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Mass();

    /// @name Vector operations
    /// @{

    /// f += factor M dx
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// addMDx(VecDeriv&,const VecDeriv&) method implemented by the component.
    virtual void addMDx(double factor = 1.0);

    /// dx = M^-1 f
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// accFromF(VecDeriv&,const VecDeriv&) method implemented by the component.
    virtual void accFromF();

    /// f += factor M dx
    ///
    /// This method must be implemented by the component.
    virtual void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0) = 0;

    /// dx = M^-1 f
    ///
    /// This method must be implemented by the component.
    virtual void accFromF(VecDeriv& a, const VecDeriv& f) = 0;

    /// Mass forces (gravity) often have null derivative
    virtual void addDForce(VecDeriv& /*df*/, const VecDeriv& /*dx*/)
    {}

    /// vMv/2 using dof->getV()
    ///
    /// This method retrieves the velocity vector and call the internal
    /// getKineticEnergy(const VecDeriv&) method implemented by the component.
    virtual double getKineticEnergy();

    virtual double getPotentialEnergy() { return 0.0;}

    /// vMv/2
    ///
    /// This method must be implemented by the component.
    virtual double getKineticEnergy( const VecDeriv& v )=0;


    /// Mat += mFact * M
    ///
    /// This method must be implemented by the component. Offset parameter gives the current Matrix block starting point.
    virtual void addMToMatrix(defaulttype::BaseMatrix * /*mat*/, double /*mFact*/, unsigned int &/*offset*/) {};

    /// This method retrieves dx vector and call the internal
    /// addMDxToVector(defaulttype::BaseVector *,const VecDeriv&, double, unsigned int&) method implemented by the component.
    virtual void addMDxToVector(defaulttype::BaseVector * resVect, double mFact, unsigned int& offset, bool dxNull);

    /// V += mFact * M * dx
    ///
    /// This method must be implemented by the component. Offset parameter gives the current Vector starting point.
    virtual void addMDxToVector(defaulttype::BaseVector * /*resVect*/, const VecDeriv * /*dx*/, double /*mFact*/, unsigned int& /*offset*/) {};

    /// initialization to export kinetic and potential energy to gnuplot files format
    virtual void initGnuplot();

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(double time);

protected:
    /// stream to export Kinematic Energy to gnuplot files
    std::ofstream* m_gnuplotFileEc;
    /// stream to export Potential Energy to gnuplot files
    std::ofstream* m_gnuplotFileEp;

    /// @}
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
