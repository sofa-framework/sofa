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
#ifndef SOFA_CORE_BEHAVIOR_MASS_H
#define SOFA_CORE_BEHAVIOR_MASS_H

#include <sofa/SofaFramework.h>
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
protected:
    Mass(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Mass();
public:
    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return this->mstate.get(); }


    /// @name Vector operations
    /// @{

    ///                         $ f += factor M dx $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// addMDx(const MechanicalParams* /* PARAMS FIRST */, DataVecDeriv&, const DataVecDeriv&, double) method implemented by the component.
    virtual void addMDx(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId fid, double factor);

    virtual void addMDx(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

    ///                            $ dx = M^-1 f $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// accFromF(VecDeriv&,const VecDeriv&) method implemented by the component.
    virtual void accFromF(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId aid);

    virtual void accFromF(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);


    /// Mass forces (gravity) often have null derivative
    virtual void addDForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv & /*df*/, const DataVecDeriv & /*dx*/ );

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
    virtual void addMBKdx(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId dfId);
    //virtual void addMBKdx(double mFactor, double bFactor, double kFactor);

    ///                         $ e = 1/2  v^t M v $
    ///
    /// This method retrieves the velocity vector and call the internal
    /// getKineticEnergy(const MechanicalParams* /* PARAMS FIRST */, const DataVecDeriv&) method implemented by the component.
    virtual double getKineticEnergy( const MechanicalParams* mparams) const;
    virtual double getKineticEnergy( const MechanicalParams* mparams /* PARAMS FIRST */, const DataVecDeriv& v) const;

    ///                         $ e = M g x $
    ///
    /// This method retrieves the positions vector and call the internal
    /// getPotentialEnergy(const MechanicalParams* /* PARAMS FIRST */, const VecCoord&) method implemented by the component.
    virtual double getPotentialEnergy( const MechanicalParams* mparams) const;
    virtual double getPotentialEnergy( const MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x  ) const;


    ///    $ m = ( Mv, cross(x,Mv)+Iw ) $
    /// linearMomentum = Mv, angularMomentum_particle = cross(x,linearMomentum), angularMomentum_body = cross(x,linearMomentum)+Iw
    ///
    /// This method retrieves the positions and velocity vectors and call the internal
    /// getMomentum(const MechanicalParams* /* PARAMS FIRST */, const VecCoord&, const VecDeriv&) method implemented by the component.
    virtual defaulttype::Vec6d getMomentum( const MechanicalParams* mparams ) const;
    virtual defaulttype::Vec6d getMomentum( const MechanicalParams*  /* PARAMS FIRST */, const DataVecCoord& , const DataVecDeriv&  ) const;



    /// @}

    /// @name Matrix operations
    /// @{

    virtual void addKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    virtual void addBToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    virtual void addMToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    /// Compute the system matrix corresponding to m M + b B + k K
    ///
    /// \param matrix matrix to add the result to
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    //virtual void addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact);
    //virtual void addMBKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double mFact, double bFact, double kFact, unsigned int &offset);

    /// addMBKToMatrix only on the subMatrixIndex
    virtual void addSubMBKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> subMatrixIndex);

    /// @}

    /// initialization to export kinetic and potential energy to gnuplot files format
    virtual void initGnuplot(const std::string path);

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, double time);

    /// perform  v += dt*g operation. Used if mass wants to added G separately from the other forces to v.
    virtual void addGravityToV(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId /*vid*/);

    virtual void addGravityToV(const MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& /* d_v */) { };

    //virtual void addGravityToV(double dt)=0;

    virtual double getElementMass(unsigned int) const;
    virtual void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

protected:
    /// stream to export Kinematic, Potential and Mechanical Energy to gnuplot files
    std::ofstream* m_gnuplotFileEnergy;
    /// @}
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_CORE)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API Mass<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Mass<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
