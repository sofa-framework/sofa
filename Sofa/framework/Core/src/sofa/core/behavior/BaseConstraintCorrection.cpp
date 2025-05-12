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
#include <sofa/core/behavior/BaseConstraintCorrection.h>

namespace sofa::core::behavior
{

BaseConstraintCorrection::BaseConstraintCorrection(){}
BaseConstraintCorrection::~BaseConstraintCorrection(){}

void BaseConstraintCorrection::rebuildSystem(SReal massFactor, SReal forceFactor)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doRebuildSystem(massFactor,forceFactor);
}
void BaseConstraintCorrection::doRebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/){}

void BaseConstraintCorrection::getComplianceWithConstraintMerge(linearalgebra::BaseMatrix* Wmerged, std::vector<int> & constraint_merge)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doGetComplianceWithConstraintMerge(Wmerged,constraint_merge);
}

void BaseConstraintCorrection::doGetComplianceWithConstraintMerge(linearalgebra::BaseMatrix*, std::vector<int>&)
{
    msg_warning() << "getComplianceWithConstraintMerge is not implemented yet " ;
}

void BaseConstraintCorrection::computeResidual(const core::ExecParams* params, linearalgebra::BaseVector * lambda)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doComputeResidual(params, lambda);
}

void BaseConstraintCorrection::doComputeResidual(const core::ExecParams* /*params*/, linearalgebra::BaseVector * /*lambda*/)
{
    dmsg_warning() << "ComputeResidual is not implemented in " << this->getName() ;
}

void BaseConstraintCorrection::getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doGetBlockDiagonalCompliance(W, begin, end);
}

void BaseConstraintCorrection::doGetBlockDiagonalCompliance(linearalgebra::BaseMatrix* /*W*/, int /*begin*/,int /*end*/)
{
    dmsg_warning() << "getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W) is not implemented in " << this->getTypeName() ;
}

bool BaseConstraintCorrection::hasConstraintNumber(int index)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doHasConstraintNumber(index);
}

bool BaseConstraintCorrection::doHasConstraintNumber(int /*index*/) {return true;}

void BaseConstraintCorrection::resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& renumbering)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doResetForUnbuiltResolution(f, renumbering);
}
void BaseConstraintCorrection::doResetForUnbuiltResolution(SReal* /*f*/, std::list<unsigned int>& /*renumbering*/) {}

void BaseConstraintCorrection::addConstraintDisplacement(SReal* d, int begin, int end)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doAddConstraintDisplacement(d, begin, end);
}
void BaseConstraintCorrection::doAddConstraintDisplacement(SReal* /*d*/, int /*begin*/, int /*end*/) {}


void BaseConstraintCorrection::setConstraintDForce(SReal* df, int begin, int end, bool update)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doSetConstraintDForce(df, begin, end, update);
}
void BaseConstraintCorrection::doSetConstraintDForce(SReal* /*df*/, int /*begin*/, int /*end*/, bool /*update*/) {}	  // f += df


void BaseConstraintCorrection::addConstraintSolver(ConstraintSolver *s)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doAddConstraintSolver(s);
}


void BaseConstraintCorrection::addComplianceInConstraintSpace(const ConstraintParams * cp, linearalgebra::BaseMatrix* W)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doAddComplianceInConstraintSpace(cp, W);
}


void BaseConstraintCorrection::getComplianceMatrix(linearalgebra::BaseMatrix* m)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doGetComplianceMatrix(m);
}

void BaseConstraintCorrection::removeConstraintSolver(ConstraintSolver *s)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doRemoveConstraintSolver(s);
}

void BaseConstraintCorrection::computeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doComputeMotionCorrectionFromLambda(cparams, dx, lambda);
}

void BaseConstraintCorrection::applyMotionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId v, MultiVecDerivId dx, ConstMultiVecDerivId correction)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doApplyMotionCorrection(cparams, x, v, dx, correction);
}

void BaseConstraintCorrection::applyPositionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId dx, ConstMultiVecDerivId correction)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doApplyPositionCorrection(cparams, x, dx, correction);
}

void BaseConstraintCorrection::applyVelocityCorrection(const ConstraintParams * cparams, MultiVecDerivId v, MultiVecDerivId dv, ConstMultiVecDerivId correction)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doApplyVelocityCorrection(cparams, v, dv, correction);
}

void BaseConstraintCorrection::applyPredictiveConstraintForce(const ConstraintParams * cparams, MultiVecDerivId f, const linearalgebra::BaseVector * lambda)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doApplyPredictiveConstraintForce(cparams, f, lambda);
}

void BaseConstraintCorrection::applyContactForce(const linearalgebra::BaseVector *f)
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doApplyContactForce(f);
}

void BaseConstraintCorrection::resetContactForce()
{
    //TODO (SPRINT SED 2025): Component state mechamism
    doResetContactForce();
}





} // namespace sofa::core::behavior

