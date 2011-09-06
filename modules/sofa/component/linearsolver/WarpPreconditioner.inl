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

// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_INL
#define SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_INL

#include <sofa/component/linearsolver/WarpPreconditioner.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/common/MechanicalMatrixVisitor.h>

#include <iostream>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class DataTypes>
WarpPreconditioner<DataTypes>::WarpPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , solverName(initData(&solverName, std::string(""), "solverName", "Name of the solver/preconditioner to warp"))
    , f_useRotationFinder(initData(&f_useRotationFinder, (unsigned)0, "useRotationFinder", "Which rotation Finder to use" ) )
{
    realSolver = NULL;
    mstate = NULL;
}


template<class DataTypes>
void WarpPreconditioner<DataTypes>::bwdInit()
{
    sofa::core::objectmodel::BaseContext * c = this->getContext();
    c->get<sofa::component::misc::BaseRotationFinder >(&rotationFinders, sofa::core::objectmodel::BaseContext::Local);

    mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    this->getContext()->get(realSolver, solverName.getValue());
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
    unsigned indRotationFinder = f_useRotationFinder.getValue()<rotationFinders.size() ? f_useRotationFinder.getValue() : 0;
    rotationFinders[indRotationFinder]->getRotations(&Rcur);

    realSolver->setSystemMBKMatrix(mparams);

    this->updateSystemMatrix();

    tmpVector1.resize(mstate->getSize()*Coord::size());
    tmpVector2.resize(mstate->getSize()*Coord::size());
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::resetSystem()
{
    if (realSolver) realSolver->resetSystem();
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::invertSystem()
{
    if (realSolver) realSolver->invertSystem();
}

/// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemRHVector(core::MultiVecDerivId v)
{
    systemRHVId = v;
}

/// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
/// This vector will be replaced by the solution of the system once solveSystem is called
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemLHVector(core::MultiVecDerivId v)
{
    systemLHVId = v;
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::updateSystemMatrix()
{
    Inherit::updateSystemMatrix();
    if (realSolver) realSolver->updateSystemMatrix();
}

/// Solve the system as constructed using the previous methods
template<class DataTypes>
void WarpPreconditioner<DataTypes>::solveSystem()
{
    if (!realSolver || !mstate) return;

    //copy : systemRHVId->tmpVector1
    executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), systemRHVId, &tmpVector1) );

    Rcur.opMulTV(&tmpVector2,&tmpVector1);

    //copy : tmpVector2->systemRHVId
    executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), systemRHVId, &tmpVector2) );

    realSolver->setSystemRHVector(systemRHVId);
    realSolver->setSystemLHVector(systemLHVId);
    realSolver->solveSystem();

    //copy : systemLHVId->tmpVector1
    executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), systemLHVId, &tmpVector1) );

    Rcur.opMulV(&tmpVector2,&tmpVector1);

    //copy : tmpVector2->systemLHVId
    executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), systemLHVId, &tmpVector2) );
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
