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
#ifndef SOFA_COMPONENT_CONSTRAINT_BASELMCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_BASELMCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/core.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 * \brief Object storing constraints base on Lagrangrian Multipliers.
 *
 *        They can be constraint on acceleration, velocity, or position.
 *        They can be grouped or individual. The resolution is then done in the OdeSolver.
 **/
class SOFA_CORE_API BaseLMConstraint: public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseLMConstraint, core::objectmodel::BaseObject);

    /// Description of the order of the constraint
    enum ConstOrder {POS,VEL,ACC};


    /**
     * \brief Expression of a line of the system created to solve the constraint
     *
     * @param idxInConstrainedDOF1 index of the line of the Jacobian in the Constrained DOF1 (can be a mapped dof)
     * @param idxInConstrainedDOF2 index of the line of the Jacobian in the Constrained DOF2 (can be a mapped dof)
     * @param correction           right hand term of the equation: corresponds to a correction we have to apply to the system
     **/
    struct ConstraintEquation
    {
        int idxInConstrainedDOF1;
        int idxInConstrainedDOF2;
        SReal correction;
    };

    /**
     * \brief Intern storage of the constraints.
     *         a ConstraintGroup is a list of equations that will be solved together.
     *  They are defined by a ConstOrder(position, velocity or acceleration)
     * @see ConstraintEquation
     * @see ConstOrder
     **/
    class ConstraintGroup
    {
        typedef sofa::helper::vector< ConstraintEquation > VecEquations;
    public:
        typedef VecEquations::const_iterator EquationConstIterator;
        typedef VecEquations::iterator       EquationIterator;

        ConstraintGroup( ConstOrder idConstraint):Order(idConstraint) {}
        /**
         * Method to add an interaction constraint to the group
         *
         * @param i0 index of the entry in the VecConst for the first object
         * @param i1 index of the entry in the VecConst for the second object
         * @param c  correction we need to apply in order to solve the constraint
         **/
        void addConstraint(  unsigned int i0, unsigned int i1, SReal c)
        {
            equations.resize(equations.size()+1);
            ConstraintEquation &eq=equations.back();
            eq.idxInConstrainedDOF1 = i0;
            eq.idxInConstrainedDOF2 = i1;
            eq.correction=c;
        }
        /**
         * Method to add a constraint to the group
         *
         * @param i  index of the entry in the VecConst for the first object
         * @param c  correction we need to apply in order to solve the constraint
         **/
        void addConstraint(  unsigned int i0,  SReal c)
        {
            equations.resize(equations.size()+1);
            ConstraintEquation &eq=equations.back();
            eq.idxInConstrainedDOF1 = i0;
            eq.idxInConstrainedDOF2 = -1; //Not used
            eq.correction=c;
        }


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
        std::size_t getNumConstraint() const { return equations.size();};

        /// Return the order of the constraint
        /// @see ConstOrder
        ConstOrder getOrder() const { return Order;};

    protected:
        /// Order of the constraint
        /// @see ConstOrder
        ConstOrder Order;
        VecEquations equations;
    };

public:
    BaseLMConstraint();

    ~BaseLMConstraint() {};


    /// Write the lines of the Jacobian
    virtual void buildJacobian()=0;
    /// Find the correspondance between num of lines in the constrained object and the simulated object
    virtual void propagateJacobian()=0;

    /// Called by MechanicalWriteLMConstaint: The Object will compute the constraints present in the current state, and create the ConstraintGroup related.
    virtual void writeConstraintEquations(ConstOrder id)=0;

    /// Gives a response impulse for a given group of constraint: This way, we can modify the Lagrange Multipliers, and handle Unilateral constraint, and more complex solutions: return a boolean indicating if the constraint group is active or not
    virtual bool LagrangeMultiplierEvaluation(SReal * /*lambda*/,core::componentmodel::behavior::BaseLMConstraint::ConstraintGroup * /*group*/) { return true;};

    /// Interface to construct a group of constraint: Giving the order of these constraints, it returns a pointer to the structure
    /// @see ConstraintGroup
    virtual ConstraintGroup* addGroupConstraint( ConstOrder Order);


    /// Get Left Hand Term for a given constraint group
    template <typename DataStorage>
    void getIndicesUsed1(const BaseLMConstraint::ConstraintGroup* group, DataStorage &used0) const
    {
        if( ! getConstrainedMechModel1() )  return;

        typedef ConstraintGroup::EquationConstIterator iterator_t;
        std::pair< iterator_t, iterator_t > range=group->data();

        for (iterator_t equation=range.first; equation!=range.second; ++equation)
        {
            if (equation->idxInConstrainedDOF1 >= 0)
            {
                line_map::const_iterator line = linesInSimulatedObject1.find(equation->idxInConstrainedDOF1);
                assert( line != linesInSimulatedObject1.end() );

                used0.push_back(line->second);

            }
        }

    }

    /// Get Left Hand Term for each ConstraintGroup of a given order
    template <typename DataStorage>
    void getIndicesUsed1(ConstOrder Order, DataStorage &used0) const
    {
        constraintOrder_t::const_iterator g = constraintOrder.find(Order);
        assert( g != constraintOrder.end() );

        const helper::vector< BaseLMConstraint::ConstraintGroup* > &constraints =
            g->second;

        for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
        {
            ConstraintGroup *group=constraints[idxGroupConstraint];
            getIndicesUsed1(group, used0);
        }
    }



    template <typename DataStorage>
    void getIndicesUsed2(const BaseLMConstraint::ConstraintGroup* group, DataStorage &used1) const
    {
        if( ! getConstrainedMechModel2() )  return;

        typedef ConstraintGroup::EquationConstIterator iterator_t;
        std::pair< iterator_t, iterator_t > range=group->data();

        for (iterator_t equation=range.first; equation!=range.second; ++equation)
        {
            if (equation->idxInConstrainedDOF2 >= 0)
            {
                typename line_map::const_iterator line =
                    linesInSimulatedObject2.find(equation->idxInConstrainedDOF2);

                assert( line != linesInSimulatedObject2.end() );

                used1.push_back(line->second);
            }
        }
    }


    template <typename DataStorage>
    void getIndicesUsed2(ConstOrder Order, DataStorage &used1) const
    {
        constraintOrder_t::const_iterator g = constraintOrder.find(Order);
        assert( g != constraintOrder.end() );

        const helper::vector< BaseLMConstraint::ConstraintGroup* > &constraints =
            g->second;

        for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
        {
            ConstraintGroup *group=constraints[idxGroupConstraint];
            getIndicesUsed2( group, used1 );
        }
    }




    /// Get Right Hand Term
    virtual void getCorrections(ConstOrder Order, helper::vector<SReal>& c);


    /// Get the internal structure: return all the constraint stored by their nature in a map
    virtual void getConstraints( std::map< ConstOrder, helper::vector< ConstraintGroup* > >  &i) { i=constraintOrder;}
    /// Get all the constraints stored of a given nature
    virtual const helper::vector< ConstraintGroup* > &getConstraintsOrder(ConstOrder Order) const
    {
        constraintOrder_t::const_iterator c = constraintOrder.find( Order );
        assert( c != constraintOrder.end());
        return c->second;
    }




    /// get the number of expressed constraints of a given order
    virtual unsigned int getNumConstraint(ConstOrder Order);


    /// get Mechanical State 1 where the constraint will be expressed (can be a Mapped mechanical state)
    virtual BaseMechanicalState* getConstrainedMechModel1() const=0;
    /// get Mechanical State 2 where the constraint will be expressed (can be a Mapped mechanical state)
    virtual BaseMechanicalState* getConstrainedMechModel2() const=0;

    /// get Mechanical State 1 where the constraint will be solved
    virtual BaseMechanicalState* getSimulatedMechModel1()const =0;
    /// get Mechanical State 2 where the constraint will be solved
    virtual BaseMechanicalState* getSimulatedMechModel2()const =0;

    /// If the constraint is applied only on a subset of particles.
    /// That way, we can optimize the time spent traversing the mappings
    /// Deactivated by default. The constraints using only a subset of particles should activate the mask,
    /// and during projectResponse(), insert the indices of the particles modified
    virtual bool useMask() {return false;}

    /// Methods to know if we have to propagate the state we want to constrain before computing the correction
    /// If the correction is computed with the simulatedDOF, there is no need, and we can reach a good speed-up
    virtual bool isCorrectionComputedWithSimulatedDOF() {return false;}

    virtual void resetConstraint();
protected:

    /// Transfer a constraint through a MechanicalMapping. Need to update the index where the equation is expressed inside the C vector
    virtual void constraintTransmissionJ1(unsigned int entry);
    virtual void constraintTransmissionJ2(unsigned int entry);

    /// Constraints stored depending on their nature
    /// @see ConstraintGroup
    typedef std::map< ConstOrder, helper::vector< ConstraintGroup* > > constraintOrder_t;
    constraintOrder_t constraintOrder;


    Data<std::string> pathObject1;
    Data<std::string> pathObject2;


    /// stores the indices of the lines in the vector C of each MechanicalState
    typedef std::map< unsigned int,unsigned int > line_map;
    line_map linesInSimulatedObject1;
    line_map linesInSimulatedObject2;
};
}
}
}
}


#endif
