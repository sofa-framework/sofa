/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_H
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_H

#include <sofa/core/core.h>
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
template<class TDataTypes>
class PairInteractionConstraint : public BaseInteractionConstraint
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(PairInteractionConstraint,TDataTypes), BaseInteractionConstraint);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;
protected:
    PairInteractionConstraint(MechanicalState<DataTypes> *mm1 = NULL, MechanicalState<DataTypes> *mm2 = NULL);

    virtual ~PairInteractionConstraint();
public:
    Data<SReal> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState1() { return mstate1; }
    BaseMechanicalState* getMechModel1() { return mstate1; }
    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState2() { return mstate2; }
    BaseMechanicalState* getMechModel2() { return mstate2; }


    using BaseConstraintSet::getConstraintViolation;
    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v);

    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param x1 and x2 are the position vectors used to compute contraint position violation
    /// \param v1 and v2 are the velocity vectors used to compute contraint velocity violation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
            , const DataVecDeriv &v1, const DataVecDeriv &v2) = 0;

    /// Construct the Jacobian Matrix
    ///
    /// \param cId is the result constraint sparse matrix Id
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex);

    /// Construct the Jacobian Matrix
    ///
    /// \param c1 and c2 are the results constraint sparse matrix
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param x1 and x2 are the position vectors used for contraint equation computation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, DataMatrixDeriv &c1, DataMatrixDeriv &c2, unsigned int &cIndex
            , const DataVecCoord &x1, const DataVecCoord &x2) = 0;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        MechanicalState<DataTypes>* mstate1 = NULL;
        MechanicalState<DataTypes>* mstate2 = NULL;
        std::string object1 = arg->getAttribute("object1","@./");
        std::string object2 = arg->getAttribute("object2","@./");
        if (object1.empty()) object1 = "@./";
        if (object2.empty()) object2 = "@./";
        context->findLinkDest(mstate1, object1, NULL);
        context->findLinkDest(mstate2, object2, NULL);

        if (!mstate1 || !mstate2)
            return false;
        return BaseInteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* p0, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::behavior::BaseInteractionConstraint::create(p0, context, arg);

        if (arg)
        {
            std::string object1 = arg->getAttribute("object1","");
            std::string object2 = arg->getAttribute("object2","");
            if (!object1.empty())
            {
                arg->setAttribute("object1", object1.c_str());
            }
            if (!object2.empty())
            {
                arg->setAttribute("object2", object2.c_str());
            }
            obj->parse(arg);
        }

        return obj;
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PairInteractionConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    SingleLink<PairInteractionConstraint<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate1;
    SingleLink<PairInteractionConstraint<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate2;

    /// Useful when the Constraint is applied only on a subset of dofs.
    /// It is automatically called by buildConstraintMatrix
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The Constraint using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
