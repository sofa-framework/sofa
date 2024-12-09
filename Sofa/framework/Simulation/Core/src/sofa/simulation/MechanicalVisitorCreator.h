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

#include <sofa/simulation/MechanicalVisitor.h>


namespace sofa::simulation
{

template<class Callable>
struct VisitorCallable
{
    Callable m_callable;
    constexpr explicit VisitorCallable(const Callable& callable) : m_callable(callable) {}
    constexpr explicit VisitorCallable(Callable&& callable) : m_callable(std::forward<Callable>(callable)) {}
};

template<class Callable, class TVisitedObject>
struct TopDownCallable : VisitorCallable<Callable>
{
    using VisitorCallable<Callable>::VisitorCallable;
    using VisitedObject = TVisitedObject;
};

template<class Callable, class TVisitedObject>
struct BottomUpCallable : VisitorCallable<Callable>
{
    using VisitorCallable<Callable>::VisitorCallable;
    using VisitedObject = TVisitedObject;
};

#define MAKE_CALLABLE(visitorCallable, Base, visitedObject) \
    template<class Callable> struct visitorCallable : Base<Callable, visitedObject> { \
    constexpr explicit visitorCallable(const Callable& callable) : Base<Callable, visitedObject>(callable) {} \
    constexpr explicit visitorCallable(Callable&& callable) : Base<Callable, visitedObject>(std::forward<Callable>(callable)) {} \
    };

#define MAKE_CREATORS(className, visitorCallable)\
    template<class Callable>\
    className<MechanicalVisitor, Callable> makeMechanicalVisitor(const sofa::core::MechanicalParams* mparams, const visitorCallable<Callable>& callable)\
    {\
        return className<MechanicalVisitor, Callable>(mparams, callable); \
    }\
    template<class Callable, class... Tail>\
    auto makeMechanicalVisitor(const sofa::core::MechanicalParams* mparams, const visitorCallable<Callable>& callable, const Tail&... tail)\
    {\
        const className<decltype(makeMechanicalVisitor(mparams, tail...)), Callable> r(mparams, callable, tail...); \
        return r;\
    }

#define MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(className, visitedObject, functionName, visitorCallable) \
    MAKE_CALLABLE(visitorCallable, TopDownCallable, visitedObject)\
    template<class Base, class Callable>\
    class className : public Base\
    {\
    public:\
        template<class... OtherCallables>\
        className(const sofa::core::MechanicalParams* mparams, const visitorCallable<Callable>& callable, const OtherCallables&... others)\
            : Base(mparams, others...), m_callable(callable.m_callable)\
        {}\
        \
        Visitor::Result functionName(Node* node, visitedObject* solver) override\
        {\
            return m_callable(node, solver);\
        }\
\
    protected:\
        Callable m_callable;\
    };\
    MAKE_CREATORS(className, visitorCallable)

#define MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(className, visitedObject, functionName, visitorCallable) \
    MAKE_CALLABLE(visitorCallable, TopDownCallable, visitedObject)\
    template<class Base, class Callable>\
    class className : public Base\
    {\
    public:\
        template<class... OtherCallables>\
        className(const sofa::core::MechanicalParams* mparams, const visitorCallable<Callable>& callable, const OtherCallables&... others)\
            : Base(mparams, others...), m_callable(callable.m_callable)\
        {}\
        \
        void functionName(Node* node, visitedObject* solver) override\
        {\
            m_callable(node, solver);\
        }\
    \
    protected:\
        Callable m_callable;\
    };\
    MAKE_CREATORS(className, visitorCallable)

MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorOdeSolver, core::behavior::OdeSolver, fwdOdeSolver, TopDownOdeSolverCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorConstraintSolver, core::behavior::ConstraintSolver, fwdConstraintSolver, TopDownConstraintSolverCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorMechanicalMapping, core::BaseMapping, fwdMechanicalMapping, TopDownMechanicalMappingCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorMappedMechanicalState, sofa::core::behavior::BaseMechanicalState, fwdMappedMechanicalState, TopDownMappedMechanicalStateCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorMechanicalState, sofa::core::behavior::BaseMechanicalState, fwdMechanicalState, TopDownMechanicalStateCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorMass, sofa::core::behavior::BaseMass, fwdMass, TopDownMassCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorForceField, sofa::core::behavior::BaseForceField, fwdForceField, TopDownForceFieldCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorInteractionForceField, sofa::core::behavior::BaseInteractionForceField, fwdInteractionForceField, TopDownInteractionForceFieldCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorProjectiveConstraintSet, sofa::core::behavior::BaseProjectiveConstraintSet, fwdProjectiveConstraintSet, TopDownProjectiveConstraintSetCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorConstraintSet, sofa::core::behavior::BaseConstraintSet, fwdConstraintSet, TopDownConstraintSetCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorInteractionProjectiveConstraintSet, sofa::core::behavior::BaseInteractionProjectiveConstraintSet, fwdInteractionProjectiveConstraintSet, TopDownInteractionProjectiveConstraintSetCallable)
MAKE_TOP_DOWN_MECHANICAL_VISITOR_TYPE(TopDownMechanicalVisitorInteractionConstraint, sofa::core::behavior::BaseInteractionConstraint, fwdInteractionConstraint, TopDownInteractionInteractionConstraintCallable)


MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorMechanicalState, core::behavior::BaseMechanicalState, bwdMechanicalState, BottomUpMechanicalStateCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorMappedMechanicalState, core::behavior::BaseMechanicalState, bwdMappedMechanicalState, BottomUpMappedMechanicalStateCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorMechanicalMapping, core::BaseMapping, bwdMechanicalMapping, BottomUpMechanicalMappingCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorOdeSolver, core::behavior::OdeSolver, bwdOdeSolver, BottomUpOdeSolverCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorConstraintSolver, core::behavior::ConstraintSolver, bwdConstraintSolver, BottomUpConstraintSolverCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorProjectiveConstraintSet, core::behavior::BaseProjectiveConstraintSet, bwdProjectiveConstraintSet, BottomUpProjectiveConstraintSetCallable)
MAKE_BOTTOM_UP_MECHANICAL_VISITOR_TYPE(BottomUpMechanicalVisitorConstraintSet, core::behavior::BaseConstraintSet, bwdConstraintSet, BottomUpConstraintSetCallable)


}
