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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_MECHANICALMATRIXVISITOR_H
#define SOFA_SIMULATION_MECHANICALMATRIXVISITOR_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif


#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseMechanicalMapping.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/InteractionForceField.h>
#include <sofa/core/behavior/InteractionConstraint.h>
#include <sofa/core/behavior/Constraint.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <iostream>

namespace sofa
{

namespace simulation
{

using std::cerr;
using std::endl;

using namespace sofa::defaulttype;
/** Base class for easily creating new actions for mechanical matrix manipulation

	During the first traversal (top-down), method processNodeTopDown(simulation::Node*) is applied to each simulation::Node. Each component attached to this node is processed using the appropriate method, prefixed by fwd.

	During the second traversal (bottom-up), method processNodeBottomUp(simulation::Node*) is applied to each simulation::Node. Each component attached to this node is processed using the appropriate method, prefixed by bwd.

	The default behavior of the fwd* and bwd* is to do nothing. Derived actions typically overload these methods to implement the desired processing.

*/
class SOFA_SIMULATION_COMMON_API MechanicalMatrixVisitor : public Visitor
{
public:
    typedef sofa::core::behavior::BaseMechanicalState::VecId VecId;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMatrixVisitor"; }

    /**@name Forward processing
    Methods called during the forward (top-down) traversal of the data structure.
     Method processNodeTopDown(simulation::Node*) calls the fwd* methods in the order given here. When there is a mapping, it is processed first, then method fwdMappedMechanicalState is applied to the BaseMechanicalState.
     When there is no mapping, the BaseMechanicalState is processed first using method fwdMechanicalState.
     Then, the other fwd* methods are applied in the given order.
        */
    ///@{

    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Process the OdeSolver
    virtual Result fwdOdeSolver(simulation::Node* /*node*/, core::behavior::OdeSolver* /*solver*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMechanicalMapping
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::behavior::BaseMechanicalMapping* /*map*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMechanicalState if it is mapped from the parent level
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
        return RESULT_PRUNE;
    }

    /// Process the BaseMechanicalState if it is not mapped from the parent level
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* /*mass*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the BaseForceField
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* /*ff*/)
    {
        return RESULT_CONTINUE;
    }


    /// Process all the InteractionForceField
    virtual Result fwdInteractionForceField(simulation::Node* node, core::behavior::InteractionForceField* ff)
    {
        return fwdForceField(node, ff);
    }

    /// Process all the BaseConstraint
    virtual Result fwdConstraint(simulation::Node* /*node*/, core::behavior::BaseConstraint* /*c*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(simulation::Node* node, core::behavior::InteractionConstraint* c)
    {
        return fwdConstraint(node, c);
    }

    ///@}

    /**@name Backward processing
    Methods called during the backward (bottom-up) traversal of the data structure.
     Method processNodeBottomUp(simulation::Node*) calls the bwd* methods.
     When there is a mapping, method bwdMappedMechanicalState is applied to the BaseMechanicalState.
     When there is no mapping, the BaseMechanicalState is processed using method bwdMechanicalState.
     Finally, the mapping (if any) is processed using method bwdMechanicalMapping.
        */
    ///@{

    /// This method calls the bwd* methods during the backward traversal. You typically do not overload it.
    virtual void processNodeBottomUp(simulation::Node* node);

    /// Process the BaseMechanicalState when it is not mapped from parent level
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {}

    /// Process the BaseMechanicalState when it is mapped from parent level
    virtual void bwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {}

    /// Process the BaseMechanicalMapping
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::behavior::BaseMechanicalMapping* /*map*/)
    {}

    /// Process the OdeSolver
    virtual void bwdOdeSolver(simulation::Node* /*node*/, core::behavior::OdeSolver* /*solver*/)
    {}

    ///@}


    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "animate";
    }

};


/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalGetMatrixDimensionVisitor : public MechanicalMatrixVisitor
{
public:
    unsigned int * const nbRow;
    unsigned int * const nbCol;
    sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalGetMatrixDimensionVisitor(unsigned int * const _nbRow, unsigned int * const _nbCol, sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL)
        : nbRow(_nbRow), nbCol(_nbCol), matrix(_matrix)
    {}

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        //ms->contributeToMatrixDimension(nbRow, nbCol);
        const unsigned int n = ms->getMatrixSize();
        if (nbRow) *nbRow += n;
        if (nbCol) *nbCol += n;
        if (matrix) matrix->addMechanicalState(ms);
        return RESULT_CONTINUE;
    }

    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::behavior::BaseMechanicalMapping* mm)
    {
        if (matrix) matrix->addMechanicalMapping(mm);
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        if (matrix) matrix->addMappedMechanicalState(ms);
        return RESULT_CONTINUE;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetMatrixDimensionVisitor"; }

};


/** Accumulate the entries of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalAddMBK_ToMatrixVisitor : public MechanicalMatrixVisitor
{
public:
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    double m, b, k;

    MechanicalAddMBK_ToMatrixVisitor(const sofa::core::behavior::MultiMatrixAccessor* _matrix, double _m=0.0, double _b=0.0, double _k=0.0)
        : matrix(_matrix),m(_m),b(_b),k(_k)
    {
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddMBK_ToMatrixVisitor"; }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*ms*/)
    {
        //ms->setOffset(offsetOnExit);
        return RESULT_CONTINUE;
    }

    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if (matrix != NULL)
        {
            ff->addMBKToMatrix(matrix,m,b,k);
        }

        return RESULT_CONTINUE;
    }

    //Masses are now added in the addMBKToMatrix call for all ForceFields

    virtual Result fwdConstraint(simulation::Node* /*node*/, core::behavior::BaseConstraint* c)
    {
        if (matrix != NULL)
        {
            c->applyConstraint(matrix);
        }

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVector2BaseVectorVisitor : public MechanicalMatrixVisitor
{
public:
    VecId src;
    BaseVector *vect;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVector2BaseVectorVisitor"; }

    MechanicalMultiVector2BaseVectorVisitor(VecId _src, defaulttype::BaseVector * _vect, const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL)
        : src(_src), vect(_vect), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (vect != NULL && offset >= 0)
        {
            unsigned int o = (unsigned int)offset;
            mm->loadInBaseVector(vect, src, o);
            offset = (int)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();
        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorPeqBaseVectorVisitor : public MechanicalMatrixVisitor
{
public:
    BaseVector *src;
    VecId dest;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorPeqBaseVectorVisitor"; }

    MechanicalMultiVectorPeqBaseVectorVisitor(VecId _dest, defaulttype::BaseVector * _src, const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL)
        : src(_src), dest(_dest), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (src!= NULL && offset >= 0)
        {
            unsigned int o = (unsigned int)offset;
            mm->addBaseVectorToState(dest, src, o);
            offset = (int)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();

        return RESULT_CONTINUE;
    }
};

} // namespace simulation

} // namespace sofa

#endif
