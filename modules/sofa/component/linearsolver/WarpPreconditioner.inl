/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/Quater.h>

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
    , f_enable(initData(&f_enable, true, "enable", "Use the preconditioner" ) )
    , f_draw_rotations_scale(initData(&f_draw_rotations_scale, 0.0, "draw_rotations_scale", "Scale to display rotations" ) )
{

    realSolver = NULL;
    mstate = NULL;

    rotationWork[0] = NULL;
    rotationWork[1] = NULL;

    indRotationFinder = -1;
    first = true;
    indexwork = 0;
}

template<class DataTypes>
WarpPreconditioner<DataTypes>::~WarpPreconditioner()
{
    if (rotationWork[0]) delete rotationWork[0];
    if (rotationWork[1]) delete rotationWork[1];

    rotationWork[0] = NULL;
    rotationWork[1] = NULL;

    delete tmpVecId;
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::bwdInit()
{
    mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    this->getContext()->get(realSolver, solverName.getValue());

    if (mstate==NULL) serr << "Error the cannot find Mstate" << sendl;
    if (realSolver==NULL) serr << "Error the cannot find the solver " << solverName.getValue() << sendl;

    sofa::core::objectmodel::BaseContext * c = this->getContext();
    c->get<sofa::core::behavior::BaseRotationFinder >(&rotationFinders, sofa::core::objectmodel::BaseContext::Local);

    sout << "Found " << rotationFinders.size() << " Rotation finders" << sendl;
    for (unsigned i=0; i<rotationFinders.size(); i++)
    {
        sout << i << " : " << rotationFinders[i]->getName() << sendl;
    }

    first = true;
    indexwork = 0;

    tmpVecId = new GraphScatteredVector(NULL,core::VecDerivId::null());

    indRotationFinder = f_useRotationFinder.getValue()<rotationFinders.size() ? f_useRotationFinder.getValue() : -1;
}

template<class DataTypes>
unsigned WarpPreconditioner<DataTypes>::getSystemDimention()
{
    return mstate->getSize()*Coord::size();
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
    realSolver->setSystemMBKMatrix(mparams);

    if (first)
    {
        first = false;
        updateSystemSize = getSystemDimention();

        if (!rotationWork[indexwork]) rotationWork[indexwork] = createRotationMatrix();
        rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
        rotationFinders[indRotationFinder]->getRotations(rotationWork[indexwork]);

        if (realSolver->isParallelSolver()) indexwork = (indexwork==0) ? 1 : 0;

        if (!rotationWork[indexwork]) rotationWork[indexwork] = createRotationMatrix();
        rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
        rotationFinders[indRotationFinder]->getRotations(rotationWork[indexwork]);
    }
    else
    {
        if (realSolver->hasUpdatedMatrix())
        {
            updateSystemSize = getSystemDimention();
            if (!rotationWork[indexwork]) rotationWork[indexwork] = createRotationMatrix();
            rotationWork[indexwork]->resize(updateSystemSize,updateSystemSize);
            rotationFinders[indRotationFinder]->getRotations(rotationWork[indexwork]);

            if (realSolver->isParallelSolver()) indexwork = (indexwork==0) ? 1 : 0;
        }
    }
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::updateSystemMatrix()
{
    realSolver->updateSystemMatrix();

    indRotationFinder = f_useRotationFinder.getValue()<rotationFinders.size() ? f_useRotationFinder.getValue() : -1;

    if (indRotationFinder>=0)
    {
        currentSystemSize = getSystemDimention();

        tmpVector1.fastResize(currentSystemSize);
        tmpVector2.fastResize(currentSystemSize);

        Rcur.resize(currentSystemSize,currentSystemSize);
        rotationFinders[indRotationFinder]->getRotations(&Rcur);
        Rcur.opMulTM(&Rcur,rotationWork[indexwork]);
    }
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemLHVector(core::MultiVecDerivId v)
{
    systemLHVId = v;
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemRHVector(core::MultiVecDerivId v)
{
    systemRHVId = v;
}

/// Solve the system as constructed using the previous methods
template<class DataTypes>
void WarpPreconditioner<DataTypes>::solveSystem()
{
    if (f_enable.getValue() && (indRotationFinder>=0))
    {
        //copy : systemRHVId->tmpVector2
        executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), systemRHVId, &tmpVector1) );

        Rcur.opMulTV(&tmpVector2,&tmpVector1);

        //copy : tmpVector1->systemRHVId
        executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), *tmpVecId, &tmpVector2) );

        realSolver->setSystemRHVector(*tmpVecId);
        realSolver->setSystemLHVector(systemLHVId);
        realSolver->solveSystem();

        //copy : systemLHVId->tmpVector1
        executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), systemLHVId, &tmpVector1) );

        Rcur.opMulV(&tmpVector2,&tmpVector1);

        //copy : tmpVector2->systemLHVId
        executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), systemLHVId, &tmpVector2) );
    }
    else
    {
        realSolver->setSystemRHVector(systemRHVId);
        realSolver->setSystemLHVector(systemLHVId);
        realSolver->solveSystem();
    }
}

/// Solve the system as constructed using the previous methods
template<class DataTypes>
bool WarpPreconditioner<DataTypes>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (J->rowSize()==0 || !realSolver) return true;

    if (indRotationFinder>=0)
    {
        JMatrixType * j_local = internalData.getLocalJ(J);
        internalData.opMulJ(&Rcur,j_local);
        return realSolver->addJMInvJt(result,j_local,fact);
    }
    else
    {
        return realSolver->addJMInvJt(result,J,fact);
    }
}

template<class DataTypes>
bool WarpPreconditioner<DataTypes>::addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (J->rowSize()==0 || !realSolver) return true;

    if (indRotationFinder>=0)
    {
        JMatrixType * j_local = internalData.getLocalJ(J);
        internalData.opMulJ(&Rcur,j_local);
        return realSolver->addMInvJt(result, j_local, fact);
    }
    else
    {
        return realSolver->addJMInvJt(result,J,fact);
    }

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

template<class DataTypes>
defaulttype::BaseMatrix* WarpPreconditioner<DataTypes>::getSystemBaseMatrix()
{
    if (realSolver) return realSolver->getSystemBaseMatrix(); else return NULL;
}

template<class DataTypes>
defaulttype::BaseVector* WarpPreconditioner<DataTypes>::getSystemRHBaseVector()
{
    if (realSolver) return realSolver->getSystemRHBaseVector(); else return NULL;
}

template<class DataTypes>
defaulttype::BaseVector* WarpPreconditioner<DataTypes>::getSystemLHBaseVector()
{
    if (realSolver) return realSolver->getSystemLHBaseVector(); else return NULL;
}

template<class DataTypes>
defaulttype::BaseMatrix* WarpPreconditioner<DataTypes>::getSystemInverseBaseMatrix()
{
    if (realSolver) return realSolver->getSystemInverseBaseMatrix(); else return NULL;
}

template<class DataTypes>
bool WarpPreconditioner<DataTypes>::readFile(std::istream& in)
{
    if (realSolver) return realSolver->readFile(in); else return false;
}

template<class DataTypes>
bool WarpPreconditioner<DataTypes>::writeFile(std::ostream& out)
{
    if (realSolver) return realSolver->writeFile(out); else return false;
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::freezeSystemMatrix()
{
    Inherit::freezeSystemMatrix();
    if (realSolver) realSolver->freezeSystemMatrix();
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{
    if (f_draw_rotations_scale.getValue() <= 0.0) return;
    if (Rcur.colSize()==0) return;
    if (Rcur.rowSize()==0) return;
    if (mstate==NULL) return;

    const VecCoord& x = *mstate->getX();

    for (int e=0; e< mstate->getSize(); e++)
    {
        sofa::defaulttype::Matrix3 RotMat;

        for (int a=0; a<3; a++)
        {
            for (int b=0; b<3; b++)
            {
                RotMat[a][b] = Rcur.element(e*3+a,e*3+b);
            }
        }

        sofa::defaulttype::Quat q;
        q.fromMatrix(RotMat);
        helper::gl::Axis::draw(DataTypes::getCPos(x[e]), q, this->f_draw_rotations_scale.getValue());
    }
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
