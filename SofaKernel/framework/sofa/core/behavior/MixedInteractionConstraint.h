/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_H
#define SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_H

#include <sofa/core/core.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing constraints between a pair of simulated body.
 *
 *  This class define the abstract API common to interaction constraints
 *  between a pair of bodies using a given type of DOFs.
 */
template<class TDataTypes1, class TDataTypes2>
class  MixedInteractionConstraint : public BaseInteractionConstraint
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(MixedInteractionConstraint,TDataTypes1,TDataTypes2), BaseInteractionConstraint);

    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::MatrixDeriv MatrixDeriv1;
    typedef typename DataTypes1::Coord Coord1;
    typedef typename DataTypes1::Deriv Deriv1;
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::MatrixDeriv MatrixDeriv2;
    typedef typename DataTypes2::Coord Coord2;
    typedef typename DataTypes2::Deriv Deriv2;

    typedef core::objectmodel::Data< VecCoord1 >		DataVecCoord1;
    typedef core::objectmodel::Data< VecDeriv1 >		DataVecDeriv1;
    typedef core::objectmodel::Data< MatrixDeriv1 >		DataMatrixDeriv1;

    typedef core::objectmodel::Data< VecCoord2 >		DataVecCoord2;
    typedef core::objectmodel::Data< VecDeriv2 >		DataVecDeriv2;
    typedef core::objectmodel::Data< MatrixDeriv2 >		DataMatrixDeriv2;
protected:
    MixedInteractionConstraint(MechanicalState<DataTypes1> *mm1 = NULL, MechanicalState<DataTypes2> *mm2 = NULL);

    virtual ~MixedInteractionConstraint();
public:
    Data<SReal> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init() override;

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes1>* getMState1() { return mstate1; }
    BaseMechanicalState* getMechModel1() override { return mstate1; }
    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes2>* getMState2() { return mstate2; }
    BaseMechanicalState* getMechModel2() override { return mstate2; }

    using BaseConstraintSet::getConstraintViolation;
    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v) override;

    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param x1 and x2 are the position vectors used to compute contraint position violation
    /// \param v1 and v2 are the velocity vectors used to compute contraint velocity violation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v, const DataVecCoord1 &x1, const DataVecCoord2 &x2
            , const DataVecDeriv1 &v1, const DataVecDeriv2 &v2) = 0;

    /// Construct the Jacobian Matrix
    ///
    /// \param cId is the result constraint sparse matrix Id
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex) override;

    /// Construct the Jacobian Matrix
    ///
    /// \param c1 and c2 are the results constraint sparse matrix
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param x1 and x2 are the position vectors used for contraint equation computation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, DataMatrixDeriv1 &c1, DataMatrixDeriv2 &c2, unsigned int &cIndex
            , const DataVecCoord1 &x1, const DataVecCoord2 &x2) = 0;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        return BaseInteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* p0, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::behavior::BaseInteractionConstraint::create(p0, context, arg);

        if (arg)
        {
            obj->parse(arg);
        }

        return obj;
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const MixedInteractionConstraint<DataTypes1,DataTypes2>* = NULL)
    {
        return DataTypes1::Name();
    }

protected:
    SingleLink<MixedInteractionConstraint<DataTypes1,DataTypes2>, MechanicalState<DataTypes1>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate1;
    SingleLink<MixedInteractionConstraint<DataTypes1,DataTypes2>, MechanicalState<DataTypes2>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate2;
    typename MechanicalState<DataTypes1>::ForceMask *mask1;
    typename MechanicalState<DataTypes2>::ForceMask *mask2;


    /// Useful when the Constraint is applied only on a subset of dofs.
    /// It is automatically called by buildConstraintMatrix
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The Constraint using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3dTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2dTypes, defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec1dTypes, defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2dTypes, defaulttype::Rigid2dTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2dTypes, defaulttype::Rigid2dTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2dTypes, defaulttype::Vec2dTypes> ;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3fTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2fTypes, defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec1fTypes, defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2fTypes, defaulttype::Rigid2fTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2fTypes, defaulttype::Rigid2fTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2fTypes, defaulttype::Vec2fTypes> ;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
