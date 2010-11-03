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


#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <iostream>

#include <sofa/core/ExecParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>

namespace sofa
{

namespace simulation
{

using std::cerr;
using std::endl;

using namespace sofa::defaulttype;
using namespace sofa::core;


/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalGetMatrixDimensionVisitor : public BaseMechanicalVisitor
{
public:
    unsigned int * const nbRow;
    unsigned int * const nbCol;
    sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalGetMatrixDimensionVisitor(unsigned int * const _nbRow, unsigned int * const _nbCol,
            sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL,
            const core::ExecParams* params = core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(params) , nbRow(_nbRow), nbCol(_nbCol), matrix(_matrix)
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

    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* mm)
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
class SOFA_SIMULATION_COMMON_API MechanicalAddMBK_ToMatrixVisitor : public MechanicalVisitor
{
public:
    const sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalAddMBK_ToMatrixVisitor(const sofa::core::behavior::MultiMatrixAccessor* _matrix, const core::MechanicalParams* mparams = core::MechanicalParams::defaultInstance() )
        : MechanicalVisitor(mparams) ,  matrix(_matrix) //,m(_m),b(_b),k(_k)
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
            ff->addMBKToMatrix(matrix,this->mparams);
        }

        return RESULT_CONTINUE;
    }

    //Masses are now added in the addMBKToMatrix call for all ForceFields

    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        if (matrix != NULL)
        {
            c->applyConstraint(matrix,this->mparams);
        }

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorToBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    ConstMultiVecId src;
    BaseVector *vect;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVector2BaseVectorVisitor"; }

    MechanicalMultiVectorToBaseVectorVisitor( ConstMultiVecId _src, defaulttype::BaseVector * _vect,
            const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL,
            const core::ExecParams* params = core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(params) , src(_src), vect(_vect), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (vect != NULL && offset >= 0)
        {
            unsigned int o = (unsigned int)offset;
            mm->copyToBaseVector(vect, src.getId(mm), o);
            offset = (int)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();
        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorPeqBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    BaseVector *src;
    MultiVecDerivId dest;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorPeqBaseVectorVisitor"; }

    MechanicalMultiVectorPeqBaseVectorVisitor(MultiVecDerivId _dest, defaulttype::BaseVector * _src,
            const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL,
            const core::ExecParams* params = core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(params) , src(_src), dest(_dest), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (src!= NULL && offset >= 0)
        {
            unsigned int o = (unsigned int)offset;
            mm->addFromBaseVectorSameSize(dest.getId(mm), src, o);
            offset = (int)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorFromBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    BaseVector *src;
    MultiVecId dest;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorPeqBaseVectorVisitor"; }

    MechanicalMultiVectorFromBaseVectorVisitor(MultiVecId _dest,
            defaulttype::BaseVector * _src,
            const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL,
            const core::ExecParams* params = core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(params) , src(_src), dest(_dest), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (src!= NULL && offset >= 0)
        {
            unsigned int o = (unsigned int)offset;
            mm->copyFromBaseVector(dest.getId(mm), src, o);
            offset = (int)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();

        return RESULT_CONTINUE;
    }
};

} // namespace simulation

} // namespace sofa

#endif
