/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace sofa
{

namespace core
{

namespace behavior
{

BaseConstraintCorrection::BaseConstraintCorrection(){}
BaseConstraintCorrection::~BaseConstraintCorrection(){}

void BaseConstraintCorrection::getComplianceWithConstraintMerge(defaulttype::BaseMatrix* /*Wmerged*/, std::vector<int> & /*constraint_merge*/)
{
    msg_warning() << "getComplianceWithConstraintMerge is not implemented yet " ;
}

void BaseConstraintCorrection::rebuildSystem(double /*massFactor*/, double /*forceFactor*/){}
void BaseConstraintCorrection::computeResidual(const core::ExecParams* /*params*/, defaulttype::BaseVector * /*lambda*/)
{
    dmsg_warning() << "ComputeResidual is not implemented in " << this->getName() ;
}

bool BaseConstraintCorrection::hasConstraintNumber(int /*index*/) {return true;}
void BaseConstraintCorrection::resetForUnbuiltResolution(double * /*f*/, std::list<unsigned int>& /*renumbering*/) {}
void BaseConstraintCorrection::addConstraintDisplacement(double * /*d*/, int /*begin*/, int /*end*/) {}
void BaseConstraintCorrection::setConstraintDForce(double * /*df*/, int /*begin*/, int /*end*/, bool /*update*/) {}	  // f += df


} // namespace behavior

} // namespace core

} // namespace sofa

