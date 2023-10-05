/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/config.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/core/behavior/PairStateAccessor.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component computing constraints between a pair of simulated body.
 *
 *  This class define the abstract API common to interaction constraints
 *  between a pair of bodies using a given type of DOFs.
 */
template<class TDataTypes>
class PairInteractionProjectiveConstraintSet : public BaseInteractionProjectiveConstraintSet, public PairStateAccessor<TDataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(PairInteractionProjectiveConstraintSet, TDataTypes), BaseInteractionProjectiveConstraintSet, SOFA_TEMPLATE2(PairStateAccessor, TDataTypes, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef objectmodel::Data<VecCoord> DataVecCoord;
    typedef objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef typename DataTypes::Real Real;
protected:
    explicit PairInteractionProjectiveConstraintSet(MechanicalState<DataTypes> *mm1 = nullptr, MechanicalState<DataTypes> *mm2 = nullptr);

    ~PairInteractionProjectiveConstraintSet() override;
public:
    Data<SReal> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    // to get rid of warnings
    using BaseInteractionProjectiveConstraintSet::projectPosition;
    using BaseInteractionProjectiveConstraintSet::projectResponse;

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method retrieves the dx vector from the MechanicalState and call
    /// the internal projectResponse(VecDeriv&,VecDeriv&) method implemented by
    /// the component.
    void projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) override;

    /// Project the L matrix of the Lagrange Multiplier equation system.
    ///
    /// This method retrieves the lines of the Jacobian Matrix from the MechanicalState and call
    /// the internal projectResponse(MatrixDeriv&) method implemented by
    /// the component.
    void projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId) override;

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method retrieves the v vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&,VecDeriv&) method implemented by
    /// the component.
    void projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId) override;

    /// Project x to constrained space (x models a position).
    ///
    /// This method retrieves the x vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&,VecCoord&) method implemented by
    /// the component.
    void projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId) override;

    /// Project dx to constrained space (dx models an acceleration).
    virtual void projectResponse(const MechanicalParams* /*mparams*/, DataVecDeriv& dx1, DataVecDeriv& dx2) = 0;

    /// Project v to constrained space (v models a velocity).
    virtual void projectVelocity(const MechanicalParams* /*mparams*/, DataVecDeriv& v1, DataVecDeriv& v2) = 0;

    /// Project x to constrained space (x models a position).
    virtual void projectPosition(const MechanicalParams* /*mparams*/, DataVecCoord& x1, DataVecCoord& x2) = 0;

    /// @}

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    void applyConstraint(const MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {

    }

    /// Project the global Mechanical Vector to constrained space using offset parameter
    void applyConstraint(const MechanicalParams* /*mparams*/, linearalgebra::BaseVector* /*vector*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {

    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T* obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        MechanicalState<DataTypes>* mstate1 = nullptr;
        MechanicalState<DataTypes>* mstate2 = nullptr;
        std::string object1 = arg->getAttribute("object1","@./");
        std::string object2 = arg->getAttribute("object2","@./");

        context->findLinkDest(mstate1, object1, nullptr);
        context->findLinkDest(mstate2, object2, nullptr);

        if (!mstate1) {
            arg->logError("Data attribute 'object1' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
            return false;
        }
        if (!mstate2) {
            arg->logError("Data attribute 'object2' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
            return false;
        }

        return BaseInteractionProjectiveConstraintSet::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* /*p0*/, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
            context->addObject(obj);

        if (arg)
        {
            const std::string object1 = arg->getAttribute("object1","");
            const std::string object2 = arg->getAttribute("object2","");
            if (!object1.empty())
            {
                arg->setAttribute("object1", object1);
            }
            if (!object2.empty())
            {
                arg->setAttribute("object2", object2);
            }
            obj->parse(arg);
        }

        return obj;
    }

    using Inherit2::getMechModel1;
    using Inherit2::getMechModel2;
};

#if !defined(SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_CPP)
extern template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<defaulttype::Rigid2Types>;


#endif

} // namespace sofa::core::behavior
