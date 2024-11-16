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

namespace sofa::core::objectmodel
{

struct DirectionalVisitor
{
    virtual ~DirectionalVisitor() = default;
    DirectionalVisitor() = default;
    virtual void operator()(sofa::core::behavior::OdeSolver*) const {}
    virtual void operator()(sofa::core::behavior::ConstraintSolver*) const {}
    virtual void operator()(sofa::core::BaseMapping*) const {}
    virtual void operator()(sofa::core::behavior::BaseMechanicalState*) const {}
    virtual void operator()(sofa::core::behavior::BaseMass*) const {}
    virtual void operator()(sofa::core::behavior::BaseForceField*) const {}
    virtual void operator()(sofa::core::behavior::BaseInteractionForceField* force) const
    {
        this->operator()((behavior::BaseForceField*)force);
    }
    virtual void operator()(sofa::core::behavior::BaseProjectiveConstraintSet*) const {}
    virtual void operator()(sofa::core::behavior::BaseConstraintSet*) const {}
    virtual void operator()(sofa::core::behavior::BaseInteractionProjectiveConstraintSet* constraint) const
    {
        this->operator()((behavior::BaseProjectiveConstraintSet*)constraint);
    }
    virtual void operator()(sofa::core::behavior::BaseInteractionConstraint* constraint) const
    {
        this->operator()((behavior::BaseConstraintSet*)constraint);
    }
};

struct SOFA_CORE_API TopDownVisitor : DirectionalVisitor
{
    TopDownVisitor() = default;
};
struct SOFA_CORE_API BottomUpVisitor : DirectionalVisitor
{
    BottomUpVisitor() = default;
};

inline TopDownVisitor topDownVisitor {};
inline BottomUpVisitor bottomUpVisitor {};


template<class VisitorDirection, class Func, class VisitedObject,
    typename = std::enable_if_t<std::is_invocable_v<Func, VisitedObject*>>>
struct SpecializedVisitor : public virtual VisitorDirection
{
    Func m_specializedFunction;

    SpecializedVisitor(const VisitorDirection& visitor, const Func& specializedFunction)
        : VisitorDirection(visitor), m_specializedFunction(specializedFunction) {}

    void operator()(VisitedObject* object) const override
    {
        m_specializedFunction(object);
    }
};

#define DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(ObjectType) \
template<class VisitorDirection, class Func, \
    typename = std::enable_if_t<std::is_invocable_v<Func, ObjectType*>>> \
SpecializedVisitor<VisitorDirection, Func, ObjectType> operator|( \
    const VisitorDirection& input, \
    const Func& f) \
{ \
    return {input, f}; \
}

DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::OdeSolver)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::ConstraintSolver)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::BaseMapping)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseMechanicalState)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseMass)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseForceField)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseInteractionForceField)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseProjectiveConstraintSet)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseConstraintSet)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseInteractionProjectiveConstraintSet)
DEFINE_PIPE_OPERATOR_FOR_OBJECT_TYPE(sofa::core::behavior::BaseInteractionConstraint)

}
