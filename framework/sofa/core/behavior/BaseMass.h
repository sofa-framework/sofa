/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_BASEMASS_H
#define SOFA_CORE_BEHAVIOR_BASEMASS_H

#include <sofa/SofaFramework.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
namespace sofa
{

namespace core
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
 *  It is often also a ForceField, computing gravity-related forces.
 */
class SOFA_CORE_API BaseMass : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseMass, objectmodel::BaseObject);
protected:
    BaseMass()
        : m_separateGravity (initData(&m_separateGravity , false, "separateGravity", "add separately gravity to velocity computation"))
        , rayleighMass (initData(&rayleighMass , SReal(0), "rayleighMass", "Rayleigh damping - mass matrix coefficient"))
    {
    }

    virtual ~BaseMass()
    {
    }
public:
    /// @name Vector operations
    /// @{

    /// f += factor M dx
    virtual void addMDx(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId fid, double factor) =0;

    /// dx = M^-1 f
    virtual void accFromF(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId aid) = 0;

    /// \brief Perform  v += dt*g operation. Used if mass wants to added G separately from the other forces to v.
    ///
    /// \param mparams \a mparams->dt() is the time step of for temporal discretization.
    virtual void addGravityToV(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId vid) = 0;

    /// vMv/2
    virtual double getKineticEnergy(const MechanicalParams* mparams = MechanicalParams::defaultInstance()) const = 0;
    /// Mgx
    virtual double getPotentialEnergy(const MechanicalParams* mparams = MechanicalParams::defaultInstance()) const = 0;

    /// (Mv,xMv+Iw) (linear and angular momenta against world origin)
    virtual defaulttype::Vec6d getMomentum(const MechanicalParams* mparams = MechanicalParams::defaultInstance()) const = 0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// \brief Add Mass contribution to global Matrix assembling.
    ///
    /// This method must be implemented by the component.
    /// \param matrix matrix to add the result to
    /// \param mparams \a mparams->mFactor() is the coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    virtual void addMToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix) = 0;

    /// @}

    /// initialization to export kinetic and potential energy to gnuplot files format
    virtual void initGnuplot(const std::string path)=0;

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, double time)=0;

    /// Get the mass relative to the DOF at \a index.
    virtual double getElementMass(unsigned int index) const =0;
    /// Get the matrix relative to the DOF at \a index.
    virtual void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const = 0;

    virtual bool isDiagonal() { return false; }

    /// Member specifying if the gravity is added separately to the DOFs velocities (in solve method),
    /// or if is added with the other forces(addForceMethod)
    Data<bool> m_separateGravity;



    /** @name Rayleigh Damping (mass contribution)
     */
    /// @{

    /// Rayleigh Damping mass matrix coefficient
    Data< SReal > rayleighMass;

    /// @}
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
