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

    unsigned int offsetOnEnter, offsetOnExit;
};


/** Compute the size of a dynamics matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalGetMatrixDimensionVisitor : public MechanicalMatrixVisitor
{
public:
    unsigned int * const nbRow;
    unsigned int * const nbCol;
    MechanicalGetMatrixDimensionVisitor(unsigned int * const _nbRow, unsigned int * const _nbCol)
        : nbRow(_nbRow), nbCol(_nbCol)
    {}

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->contributeToMatrixDimension(nbRow, nbCol);
        return RESULT_CONTINUE;
    }
    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetMatrixDimensionVisitor"; }

};


/** Accumulate the entries of a dynamics matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalAddMBK_ToMatrixVisitor : public MechanicalMatrixVisitor
{
public:
    BaseMatrix *mat;
    double m, b, k;
    //    unsigned int offset, offsetBckUp;

    MechanicalAddMBK_ToMatrixVisitor(BaseMatrix *_mat, double _m=0.0, double _b=0.0, double _k=0.0, unsigned int _offset=0)
        : mat(_mat),m(_m),b(_b),k(_k)
    {
        offsetOnEnter = _offset;
        offsetOnExit = _offset;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddMBK_ToMatrixVisitor"; }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->setOffset(offsetOnExit);
        return RESULT_CONTINUE;
    }

    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if ((mat != NULL)&&(k!=0.0))
        {
            //offsetOnExit = offsetOnEnter;
            ff->addMBKToMatrix(mat,m,b,k,offsetOnEnter);
        }

        return RESULT_CONTINUE;
    }

    //Masses are now added in the addMBKToMatrix call for all ForceFields
    /*
    virtual Result fwdMass(simulation::Node*, core::behavior::BaseMass* mass)
    {
    if ((mat != NULL)&&(m!=0.0))
      {
        //offsetOnExit = offsetOnEnter;
        mass->addMToMatrix(mat,m,offsetOnEnter);
      }

           return RESULT_CONTINUE;
         }
         */

    virtual Result fwdConstraint(simulation::Node* /*node*/, core::behavior::BaseConstraint* c)
    {
        if (mat != NULL)
        {
            //offsetOnExit = offsetOnEnter;
            c->applyConstraint(mat, offsetOnEnter);
        }

        return RESULT_CONTINUE;
    }
};

#if 0 // deprecated: as dx is stored in MechanicalState, compute df there and then convert to BaseVector using MechanicalMultiVector2BaseVectorVisitor
/** Accumulate the entries of a dynamics vector (e.g. force) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalAddMBKdx_ToVectorVisitor : public MechanicalMatrixVisitor
{
public:
    BaseVector *vect;
    VecId dx;
    double m, b, k;
    //    unsigned int offset, offsetBckUp;

    MechanicalAddMBKdx_ToVectorVisitor(BaseVector *_vect, VecId _dx, double _m=0.0, double _b=0.0, double _k=0.0, unsigned int _offset=0)
        : vect(_vect), dx(_dx), m(_m),b(_b),k(_k)
    {
        offsetOnEnter = _offset;
        offsetOnExit = _offset;
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->setOffset(offsetOnExit);

        if (!dx.isNull())
            mm->setDx(dx);
        return RESULT_CONTINUE;
    }

    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if ((vect != NULL)&&(k != 0.0))
        {
            //	offsetOnExit = offsetOnEnter;
            ff->addKDxToVector(vect,k,offsetOnEnter);
        }

        return RESULT_CONTINUE;
    }

    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass)
    {
        if ((vect != NULL)&&(m != 0.0))
        {
            //	offsetOnExit = offsetOnEnter;
            if (dx.isNull())
                std::cout << "Dx Null\n";
            else
                std::cout << "Dx Not Null\n";

            mass->addMDxToVector(vect,m,offsetOnEnter,dx.isNull());
        }

        return RESULT_CONTINUE;
    }

    virtual Result fwdConstraint(simulation::Node* /*node*/, core::behavior::BaseConstraint* c)
    {
        if (vect != NULL)
        {
            //	offsetOnExit = offsetOnEnter;
            c->applyConstraint(vect, offsetOnEnter);
        }

        return RESULT_CONTINUE;
    }
};
#endif

class SOFA_SIMULATION_COMMON_API MechanicalMultiVector2BaseVectorVisitor : public MechanicalMatrixVisitor
{
public:
    VecId src;
    BaseVector *vect;
    unsigned int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVector2BaseVectorVisitor"; }

    MechanicalMultiVector2BaseVectorVisitor(VecId _src, defaulttype::BaseVector * _vect, unsigned int _offset=0)
        : src(_src),vect(_vect),offset(_offset)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (vect!= NULL)
        {
            mm->loadInBaseVector(vect, src, offset);
        }

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorPeqBaseVectorVisitor : public MechanicalMatrixVisitor
{
public:
    BaseVector *src;
    VecId dest;
    unsigned int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorPeqBaseVectorVisitor"; }

    MechanicalMultiVectorPeqBaseVectorVisitor(VecId _dest, defaulttype::BaseVector * _src, unsigned int _offset=0)
        : src(_src),dest(_dest),offset(_offset)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (src!= NULL)
        {
            mm->addBaseVectorToState(dest, src, offset);
        }

        return RESULT_CONTINUE;
    }
};

} // namespace simulation

} // namespace sofa

#endif
