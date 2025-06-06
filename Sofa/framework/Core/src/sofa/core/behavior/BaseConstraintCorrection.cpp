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

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa::core::behavior
{

BaseConstraintCorrection::BaseConstraintCorrection(){}
BaseConstraintCorrection::~BaseConstraintCorrection() {}

void BaseConstraintCorrection::rebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/){}

void BaseConstraintCorrection::getComplianceWithConstraintMerge(linearalgebra::BaseMatrix* /*Wmerged*/, std::vector<int> & /*constraint_merge*/)
{
    msg_warning() << "getComplianceWithConstraintMerge is not implemented yet " ;
}

void BaseConstraintCorrection::computeResidual(const core::ExecParams* /*params*/, linearalgebra::BaseVector * /*lambda*/)
{
    dmsg_warning() << "ComputeResidual is not implemented in " << this->getName() ;
}

void BaseConstraintCorrection::getBlockDiagonalCompliance(linearalgebra::BaseMatrix* /*W*/, int /*begin*/,int /*end*/)
{
    dmsg_warning() << "getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W) is not implemented in " << this->getTypeName() ;
}

bool BaseConstraintCorrection::hasConstraintNumber(int /*index*/) {return true;}
void BaseConstraintCorrection::resetForUnbuiltResolution(SReal* /*f*/, std::list<unsigned int>& /*renumbering*/) {}
void BaseConstraintCorrection::addConstraintDisplacement(SReal* /*d*/, int /*begin*/, int /*end*/) {}
void BaseConstraintCorrection::setConstraintDForce(SReal* /*df*/, int /*begin*/, int /*end*/, bool /*update*/) {}	  // f += df

SReal BaseConstraintCorrection::correctionFactor(const sofa::core::behavior::OdeSolver* solver, const ConstraintOrder& constraintOrder)
{
    if (solver)
    {
        switch (constraintOrder)
        {
            case core::ConstraintOrder::POS_AND_VEL :
            case core::ConstraintOrder::POS :
                return solver->getPositionIntegrationFactor();

            case core::ConstraintOrder::ACC :
            case core::ConstraintOrder::VEL :
                return solver->getVelocityIntegrationFactor();

            default :
                break;
        }
    }

    return 1.0_sreal;
}

} // namespace sofa::core::behavior

