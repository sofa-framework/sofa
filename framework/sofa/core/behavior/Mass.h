/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MASS_H
#define SOFA_CORE_BEHAVIOR_MASS_H

#include <sofa/core/core.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

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
 *  It is also a ForceField, computing gravity-related forces.
 */
template<class DataTypes>
class Mass : virtual public ForceField<DataTypes>, public BaseMass
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(Mass,DataTypes), SOFA_TEMPLATE(ForceField,DataTypes), BaseMass);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

    Mass(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Mass();

    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return this->mstate; }


    /// @name Vector operations
    /// @{

    ///                         $ f += factor M dx $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// addMDx(DataVecDeriv&, const DataVecDeriv&, double, const MechanicalParams*) method implemented by the component.
    virtual void addMDx(MultiVecDerivId fid, double factor, const MechanicalParams* mparams);

    virtual void addMDx(DataVecDeriv& f, const DataVecDeriv& dx, double factor, const MechanicalParams* mparams);

    /// @deprecated
    virtual void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);


    ///                            $ dx = M^-1 f $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// accFromF(VecDeriv&,const VecDeriv&) method implemented by the component.
    virtual void accFromF(MultiVecDerivId aid, const MechanicalParams* mparams);

    virtual void accFromF(DataVecDeriv& a, const DataVecDeriv& f, const MechanicalParams* mparams);

    /// @deprecated
    virtual void accFromF(VecDeriv& a, const VecDeriv& f);


    /// Mass forces (gravity) often have null derivative
    virtual void addDForce(DataVecDeriv & /*df*/, const DataVecDeriv & /*dx*/ , const MechanicalParams* /*mparams*/ );

    /// Accumulate the contribution of M, B, and/or K matrices multiplied
    /// by the dx vector with the given coefficients.
    ///
    /// This method computes
    /// $ df += mFactor M dx + bFactor B dx + kFactor K dx $
    /// For masses, it calls both addMdx and addDForce (which is often empty).
    ///
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKdx(MultiVecDerivId dfId , const MechanicalParams* mparams);
    //virtual void addMBKdx(double mFactor, double bFactor, double kFactor);

    ///                         $ e = 1/2  v^t M v $
    ///
    /// This method retrieves the velocity vector and call the internal
    /// getKineticEnergy(const DataVecDeriv&, const MechanicalParams*) method implemented by the component.
    virtual double getKineticEnergy( const MechanicalParams* mparams) const;
    virtual double getKineticEnergy( const DataVecDeriv& v   , const MechanicalParams* mparams) const;

    /// @deprecated
    virtual double getKineticEnergy( const VecDeriv& v       ) const;

    ///                         $ e = M g x $
    ///
    /// This method retrieves the positions vector and call the internal
    /// getPotentialEnergy(const VecCoord&, const MechanicalParams*) method implemented by the component.
    virtual double getPotentialEnergy( const MechanicalParams* mparams) const;
    virtual double getPotentialEnergy( const DataVecCoord& x , const MechanicalParams* mparams  ) const;

    /// @deprecated
    virtual double getPotentialEnergy( const VecCoord& x       ) const;

    /// @}

    /// @name Matrix operations
    /// @{

    /// @deprecated
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const MechanicalParams* mparams);
    //virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset);

    /// @deprecated
    virtual void addMToMatrix(sofa::defaulttype::BaseMatrix * matrix, double mFact, unsigned int &offset);
    virtual void addMToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const MechanicalParams* mparams);

    /// Compute the system matrix corresponding to m M + b B + k K
    ///
    /// \param matrix matrix to add the result to
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const MechanicalParams* mparams);
    //virtual void addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact);
    //virtual void addMBKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double mFact, double bFact, double kFact, unsigned int &offset);

    /// @}

    /// initialization to export kinetic and potential energy to gnuplot files format
    virtual void initGnuplot(const std::string path);

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(double time, const MechanicalParams* mparams = MechanicalParams::defaultInstance());

    /// perform  v += dt*g operation. Used if mass wants to added G separately from the other forces to v.
    virtual void addGravityToV(MultiVecDerivId /*vid*/, const MechanicalParams* /*mparams*/) {}
    //virtual void addGravityToV(double dt)=0;

    virtual double getElementMass(unsigned int) const;
    virtual void   getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

protected:
    /// stream to export Kinematic, Potential and Mechanical Energy to gnuplot files
    std::ofstream* m_gnuplotFileEnergy;
    /// @}
};


/** Return the inertia force applied to a body referenced in a moving coordinate system.
 *  \param sv spatial velocity (omega, vorigin) of the coordinate system
 *  \param a acceleration of the origin of the coordinate system
 *  \param m mass of the body
 *  \param x position of the body in the moving coordinate system
 *  \param v velocity of the body in the moving coordinate system
 *         This default implementation returns no inertia.
 */
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ )
{
    return Deriv();
    //const Deriv& omega=sv.getAngularVelocity();
    //return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)
extern template class SOFA_CORE_API Mass<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid2dTypes>;

extern template class SOFA_CORE_API Mass<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid2fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
