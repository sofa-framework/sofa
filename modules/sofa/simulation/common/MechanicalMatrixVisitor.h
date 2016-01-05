/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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


/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalGetMatrixDimensionVisitor : public BaseMechanicalVisitor
{
public:
    size_t * const nbRow;
    size_t * const nbCol;
    sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalGetMatrixDimensionVisitor(
        const core::ExecParams* params, size_t * const _nbRow, size_t * const _nbCol,
        sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL )
        : BaseMechanicalVisitor(params) , nbRow(_nbRow), nbCol(_nbCol), matrix(_matrix)
    {}

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        //ms->contributeToMatrixDimension(nbRow, nbCol);
        const unsigned int n = (unsigned int)ms->getMatrixSize();
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

/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalGetConstraintJacobianVisitor : public BaseMechanicalVisitor
{
public:
    defaulttype::BaseMatrix * J;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    ptrdiff_t offset;

    MechanicalGetConstraintJacobianVisitor(
        const core::ExecParams* params, defaulttype::BaseMatrix * _J, const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL)
        : BaseMechanicalVisitor(params) , J(_J), matrix(_matrix), offset(0)
    {}

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        if (matrix) offset = matrix->getGlobalOffset(ms);

        size_t o = (size_t)offset;
        ms->getConstraintJacobian(this->params,J,o);
        offset = (ptrdiff_t)o;
        return RESULT_CONTINUE;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetConstraintJacobianVisitor"; }
};

/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalIntegrateConstraintsVisitor : public BaseMechanicalVisitor
{
public:


    const sofa::defaulttype::BaseVector *src;
    const double positionFactor;// use the OdeSolver to get the position integration factor
    const double velocityFactor;// use the OdeSolver to get the position integration factor
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    ptrdiff_t offset;

    MechanicalIntegrateConstraintsVisitor(
        const core::ExecParams* params, defaulttype::BaseVector * _src, double pf,double vf, const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL)
        : BaseMechanicalVisitor(params) , src(_src), positionFactor(pf), velocityFactor(vf), matrix(_matrix), offset(0)
    {}

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        if (matrix) offset = matrix->getGlobalOffset(ms);
        if (src!= NULL && offset >= 0)
        {
            size_t o = (size_t)offset;
            ms->copyFromBaseVector(core::VecDerivId::dx(), src, o);
            offset = (ptrdiff_t)o;

            //x = x_free + dx * positionFactor;
            ms->vOp(params,core::VecCoordId::position(),core::ConstVecCoordId::freePosition(),core::VecDerivId::dx(),positionFactor);

            //v = v_free + dx * velocityFactor;
            ms->vOp(params,core::VecDerivId::velocity(),core::ConstVecDerivId::freeVelocity(),core::VecDerivId::dx(),velocityFactor);

            //dx *= positionFactor;
            ms->vOp(params,core::VecDerivId::dx(),core::VecDerivId::null(),core::ConstVecDerivId::dx(),positionFactor);

            //force = 0
//            ms->vOp(params,core::VecDerivId::force(),core::ConstVecDerivId::null(),core::VecDerivId::null(),0.0);
        }

        return RESULT_CONTINUE;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalIntegrateConstraintsVisitor"; }
};



/** Accumulate the entries of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_COMMON_API MechanicalAddMBK_ToMatrixVisitor : public MechanicalVisitor
{
public:
    const sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalAddMBK_ToMatrixVisitor(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* _matrix )
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
            assert( !ff->isCompliance.getValue() ); // if one day this visitor has to be used with compliance, K from compliance should not be added (by tweaking mparams with kfactor=0)
            ff->addMBKToMatrix(this->mparams, matrix);
        }

        return RESULT_CONTINUE;
    }

    //Masses are now added in the addMBKToMatrix call for all ForceFields

    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        if (matrix != NULL)
        {
            c->applyConstraint(this->mparams, matrix);
        }

        return RESULT_CONTINUE;
    }
};

/** Accumulate the entries of a mechanical matrix (mass or stiffness) of the whole scene ONLY ON THE subMatrixIndex */
class SOFA_SIMULATION_COMMON_API MechanicalAddSubMBK_ToMatrixVisitor : public MechanicalVisitor
{
public:
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    const helper::vector<size_t> & subMatrixIndex; // index of the point where the matrix must be computed

    MechanicalAddSubMBK_ToMatrixVisitor(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* _matrix, const helper::vector<size_t> & Id)
        : MechanicalVisitor(mparams) ,  matrix(_matrix), subMatrixIndex(Id) //,m(_m),b(_b),k(_k)
    {
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddSubMBK_ToMatrixVisitor"; }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*ms*/)
    {
        //ms->setOffset(offsetOnExit);
        return RESULT_CONTINUE;
    }

    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if (matrix != NULL)
        {
            assert( !ff->isCompliance.getValue() ); // if one day this visitor has to be used with compliance, K from compliance should not be added (by tweaking mparams with kfactor=0)
            ff->addSubMBKToMatrix(this->mparams, matrix, subMatrixIndex);
        }

        return RESULT_CONTINUE;
    }

    //Masses are now added in the addMBKToMatrix call for all ForceFields

    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        if (matrix != NULL)
        {
            c->applyConstraint(this->mparams, matrix);
        }

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorToBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::ConstMultiVecId src;
    sofa::defaulttype::BaseVector *vect;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVector2BaseVectorVisitor"; }

    MechanicalMultiVectorToBaseVectorVisitor(
        const core::ExecParams* params,
        sofa::core::ConstMultiVecId _src, defaulttype::BaseVector * _vect,
        const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL )
        : BaseMechanicalVisitor(params) , src(_src), vect(_vect), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (vect != NULL && offset >= 0)
        {
            size_t o = (size_t)offset;
            mm->copyToBaseVector(vect, src.getId(mm), o);
            offset = (ptrdiff_t)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();
        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorPeqBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    sofa::defaulttype::BaseVector *src;
    sofa::core::MultiVecDerivId dest;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorPeqBaseVectorVisitor"; }

    MechanicalMultiVectorPeqBaseVectorVisitor(
        const core::ExecParams* params, sofa::core::MultiVecDerivId _dest, defaulttype::BaseVector * _src,
        const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL )
        : BaseMechanicalVisitor(params) , src(_src), dest(_dest), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (src!= NULL && offset >= 0)
        {
            size_t o = (size_t)offset;
            mm->addFromBaseVectorSameSize(dest.getId(mm), src, o);
            offset = (ptrdiff_t)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();

        return RESULT_CONTINUE;
    }
};

class SOFA_SIMULATION_COMMON_API MechanicalMultiVectorFromBaseVectorVisitor : public BaseMechanicalVisitor
{
public:
    sofa::defaulttype::BaseVector *src;
    sofa::core::MultiVecId dest;
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    int offset;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalMultiVectorFromBaseVectorVisitor"; }

    MechanicalMultiVectorFromBaseVectorVisitor(
        const core::ExecParams* params, sofa::core::MultiVecId _dest,
        defaulttype::BaseVector * _src,
        const sofa::core::behavior::MultiMatrixAccessor* _matrix = NULL )
        : BaseMechanicalVisitor(params) , src(_src), dest(_dest), matrix(_matrix), offset(0)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        if (matrix) offset = matrix->getGlobalOffset(mm);
        if (src!= NULL && offset >= 0)
        {
            size_t o = (size_t)offset;
            mm->copyFromBaseVector(dest.getId(mm), src, o);
            offset = (ptrdiff_t)o;
        }
        //if (!matrix) offset += mm->getMatrixSize();

        return RESULT_CONTINUE;
    }
};

} // namespace simulation

} // namespace sofa

#endif
