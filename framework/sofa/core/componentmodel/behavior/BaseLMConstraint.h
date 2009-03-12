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
    /// Description of the nature of the constraint
    enum ConstId {POS,VEL,ACC};

    /// Right hand term creation.
    enum ValueId
    {
        FINAL      ///Desired value of the constraint
        , FACTOR     ///when we want a given factor of a state, to be as expected value of a constraint
        , CORRECTION ///Correction to apply, in order to satisfy the constraint
    };

    /**
     * \brief Intern storage of the constraints.
     *         a constraintGroup is a list of constraint that will be solved together.
     *
     *  They are defined by a ConstId(position, velocity or acceleration), indices corresponding of the entries in the VecConst vector, value needed to compute the right hand term
     **/
    class constraintGroup
    {
    public:
        constraintGroup( ConstId typeConstraint):Id(typeConstraint) {}
        /**
         * Method to add a constraint to the group
         *
         * @param i0 index of the entry in the VecConst for the first object
         * @param i1 index of the entry in the VecConst for the second object
         * @param value term to compute the right hand term of the matrix
         * @param t the way of computing the right hand term
         * @see ValueId
         **/
        void addConstraint(  unsigned int i0, unsigned int i1, double value, ValueId t)
        {
            index[0].push_back(i0); index[1].push_back(i1);
            expectedValue.push_back(value);
            typeValue.push_back(t);
        }
        /**
         * Method to retrieve one of the constraint in the group
         *
         * @param i index of constraint in this group
         * @param indexVecConst0 index of the entry in the VecConst for the first object
         * @param indexVecConst1 index of the entry in the VecConst for the second object
         * @param value term to compute the right hand term of the matrix
         * @param t the way of computing the right hand term
         **/
        void getConstraint(const unsigned int i,
                unsigned int &indexVecConst0, unsigned int &indexVecConst1,double &value, ValueId &t) const
        {
            indexVecConst0 = index[0][i]; indexVecConst1 = index[1][i];
            value = expectedValue[i];
            t     = typeValue[i];
        }

        /// Retrieves only the indices in the VecConst for a given constraint of the group
        void   getIndices          (const unsigned int entry, unsigned int &i0, unsigned int &i1) const {i0=index[0][entry]; i1=index[1][entry];}
        /// Retrieves only the value needed to compute the right hand term for a specific constraint of the group
        double getExpectedValue    (const unsigned int entry) const {return expectedValue[entry];}
        /// Retrieves only the way to compute the right hand term for a specific cosntraint of the group
        double getExpectedValueType(const unsigned int entry) const {return typeValue[entry];}

        ///Retrieves all the indices in the VecConst for the first object
        std::vector< unsigned int > getIndicesUsed0()         const {return index[0];}
        ///Retrieves all the indices in the VecConst for the second object
        std::vector< unsigned int > getIndicesUsed1()         const {return index[1];}
        ///Retrieves all the values needed to compute the right hand term
        std::vector< double >       getExpectedValues()       const {return expectedValue;}
        std::vector< ValueId >      getExpectedValuesType()   const {return typeValue;}


        /// Return the number of constraint contained in this group
        std::size_t getNumConstraint() const { return expectedValue.size();};

        /// Return the nature of the constraint
        /// @see ConstId
        ConstId getId() const { return Id;};


    protected:
        /// Nature of the constraint
        /// @see ConstId
        ConstId Id;
        /// Indices of the entries in the VecConst for the two objects
        std::vector< unsigned int > index[2];
        /// Value to compute the right hand term
        std::vector< double > expectedValue;
        /// Way to compute the right hand term
        /// @see ValueId
        std::vector< ValueId > typeValue;

    };

public:
    BaseLMConstraint();

    ~BaseLMConstraint() {};

    /// Called by MechanicalAccumulateLMConstaint: The Object will compute the constraints present in the current state, and create the constraintGroup related.
    virtual void writeConstraintEquations()=0;

    /// Interface to construct a group of constraint: Giving the nature of these constraints, it returns a pointer to the structure
    /// @see constraintGroup
    virtual constraintGroup* addGroupConstraint( ConstId Id);

    /// Get the internal structure: return all the constraint stored by their nature in a map
    virtual void getConstraints(std::map< ConstId, std::vector< constraintGroup > >  &i) { i=constraintId;}
    /// Get all the constraints stored of a given nature
    virtual void getConstraintsId(ConstId Id, std::vector< constraintGroup > &i ) { i=constraintId[Id];}


    virtual void getIndicesUsed(ConstId Id, std::vector< unsigned int > &used0,std::vector< unsigned int > &used1);
    virtual void getExpectedValues(ConstId Id, std::vector< double > &expected);
    virtual void getExpectedValuesType(ConstId Id, std::vector< ValueId > &t);


    virtual BaseMechanicalState* getMechModel1()=0;
    virtual BaseMechanicalState* getMechModel2()=0;

    virtual unsigned int getNumConstraint(ConstId Id);
    virtual double getError() {return 0;}

    virtual void clear() {constraintId.clear();}
protected:
    Data<std::string> pathObject1;
    Data<std::string> pathObject2;

    /// Constraints stored depending on their nature
    /// @see constraintGroup
    std::map< ConstId, std::vector< constraintGroup > > constraintId;
};
}
}
}
}


#endif
