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
#ifndef SOFA_COMPONENT_CONSTRAINT_BASELMCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_BASELMCONSTRAINT_H

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/core.h>

namespace sofa
{

namespace core
{

namespace behavior
{




/**
 * \brief Expression of a line of the system created to solve the constraint
*
* @param idx                  index of the equation in the constraint equation system stored within the mechanical states
* @param correction           right hand term of the equation: corresponds to a correction we have to apply to the system
* @param constraintId         actual index of the line corresponding to the constraint equation in the whole system: can be different from idx
**/
struct  SOFA_CORE_API ConstraintEquation
{
    int idx;
    SReal correction;
    unsigned int constraintId;
};





/**
 * \brief Intern storage of the constraints.
 *         a ConstraintGroup is a list of equations that will be solved together.
 *  They are defined by a ConstOrder(position, velocity or acceleration)
 * @see ConstraintEquation
 * @see ConstOrder
 **/
class SOFA_CORE_API  ConstraintGroup
{
    typedef sofa::helper::vector< ConstraintEquation > VecEquations;
public:
    typedef VecEquations::const_iterator EquationConstIterator;
    typedef VecEquations::iterator       EquationIterator;

    ConstraintGroup(ConstraintParams::ConstOrder idConstraint);
    /**
     * Method to add an interaction constraint to the group
     *
    * @param idx index of the equation
     * @param c  correction we need to apply in order to solve the constraint
     **/
    void addConstraint( unsigned int &constraintId, unsigned int idx, SReal c);



    /// Random Access to an equation
    const ConstraintEquation &getConstraint(const unsigned int i) const
    {
        EquationConstIterator it=equations.begin();
        std::advance(it,i);
        return *it;
    }

    ConstraintEquation &getConstraint(const unsigned int i)
    {
        EquationIterator it=equations.begin();
        std::advance(it,i);
        return *it;
    }


    /// Retrieve all the equations
    std::pair< EquationConstIterator,EquationConstIterator> data() const
    {
        return std::make_pair( equations.begin(), equations.end());
    }

    std::pair< EquationIterator,EquationIterator > data()
    {
        return std::make_pair( equations.begin(), equations.end());
    }


    /// Return the number of constraint contained in this group
    std::size_t getNumConstraint() const { return equations.size();}

    /// Return the order of the constraint
    /// @see ConstOrder
    ConstraintParams::ConstOrder getOrder() const { return Order;}

    bool isActive()const {return active;}
    void setActive(bool b) {active=b;}
protected:
    /// Order of the constraint
    /// @see ConstOrder
    ConstraintParams::ConstOrder Order;
    VecEquations equations;
    bool active;
};


/**
 * \brief Object storing constraints base on Lagrangrian Multipliers.
 *
 *        They can be constraint on acceleration, velocity, or position.
 *        They can be grouped or individual. The resolution is then done in the OdeSolver.
 **/
class SOFA_CORE_API BaseLMConstraint: public BaseConstraintSet
{
public:
    SOFA_ABSTRACT_CLASS(BaseLMConstraint, BaseConstraintSet);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseLMConstraint)

protected:
    BaseLMConstraint();

    ~BaseLMConstraint() {}
public:

    /// Called by MechanicalWriteLMConstaint: The Object will compute the constraints present in the current state, and create the ConstraintGroup related.
    virtual void writeConstraintEquations(unsigned int& lineNumber, MultiVecId id, ConstraintParams::ConstOrder order)=0;

    /// Compute the new Lagrange Multiplier given a block of the compliance matrix W, and the current correction (left hand term) and previous Lagrange Multiplier
    virtual void LagrangeMultiplierEvaluation(const SReal* /*W*/,
            const SReal* /*c*/, SReal* /*Lambda*/,
            ConstraintGroup * /*group*/) {}


    /// Get Right Hand Term
    virtual void getConstraintViolation(const sofa::core::ConstraintParams*, defaulttype::BaseVector * /*v*/ );

    using BaseConstraintSet::getConstraintViolation;
    // Override used in LMConstraintSolver::buildSystem method
    void getConstraintViolation(defaulttype::BaseVector *v, const core::ConstraintParams::ConstOrder );


    /// Get the internal structure: return all the constraint stored by their nature in a map
    virtual void getConstraints( std::map< ConstraintParams::ConstOrder, helper::vector< ConstraintGroup* > >  &i) { i=constraintOrder;}
    /// Get all the constraints stored of a given nature
    virtual const helper::vector< ConstraintGroup* > &getConstraintsOrder(ConstraintParams::ConstOrder Order) const
    {
        constraintOrder_t::const_iterator c = constraintOrder.find( Order );
        assert( c != constraintOrder.end());
        return c->second;
    }


    /// Get Left Hand Term for a given constraint group
    template <typename DataStorage>
    void getEquationsUsed(const ConstraintGroup* group, DataStorage &used0) const
    {
        typedef ConstraintGroup::EquationConstIterator iterator_t;
        std::pair< iterator_t, iterator_t > range=group->data();
        for (iterator_t equation=range.first; equation!=range.second; ++equation) used0.push_back(equation->idx);
    }

    /// Get Left Hand Term for each ConstraintGroup of a given order
    template <typename DataStorage>
    void getEquationsUsed(ConstraintParams::ConstOrder Order, DataStorage &used0) const
    {
        constraintOrder_t::const_iterator g = constraintOrder.find(Order);
        if (g == constraintOrder.end()) return;

        const helper::vector< ConstraintGroup* > &constraints = g->second;
        for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
        {
            ConstraintGroup *group=constraints[idxGroupConstraint];
            getEquationsUsed(group, used0);
        }
    }



    /// get the number of expressed constraints of a given order
    virtual unsigned int getNumConstraint(ConstraintParams::ConstOrder Order);


    /// get Mechanical State 1 where the constraint will be expressed (can be a Mapped mechanical state)
    virtual BaseMechanicalState* getConstrainedMechModel1() const=0;
    /// get Mechanical State 2 where the constraint will be expressed (can be a Mapped mechanical state)
    virtual BaseMechanicalState* getConstrainedMechModel2() const=0;

    /// get Mechanical State 1 where the constraint will be solved
    virtual BaseMechanicalState* getSimulatedMechModel1()const =0;
    /// get Mechanical State 2 where the constraint will b*e solved
    virtual BaseMechanicalState* getSimulatedMechModel2()const =0;

    /// Useful when the Constraint is applied only on a subset of dofs.
    /// It is automatically called by ???
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The Constraint using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() = 0;

    /// Methods to know if we have to propagate the state we want to constrain before computing the correction
    /// If the correction is computed with the simulatedDOF, there is no need, and we can reach a good speed-up
    virtual bool isCorrectionComputedWithSimulatedDOF(ConstraintParams::ConstOrder) const {return false;}

    virtual void resetConstraint();
protected:

    /// Interface to construct a group of constraint: Giving the order of these constraints, it returns a pointer to the structure
    /// @see ConstraintGroup
    virtual ConstraintGroup* addGroupConstraint(ConstraintParams::ConstOrder Order);

    /// Constraints stored depending on their nature
    /// @see ConstraintGroup
    typedef std::map< ConstraintParams::ConstOrder, helper::vector< ConstraintGroup* > > constraintOrder_t;
    constraintOrder_t constraintOrder;

    Data<std::string> pathObject1;
    Data<std::string> pathObject2;
};

} // namespace behavior

} // namespace core

} // namespace sofa


#endif
