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

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PMLBody.h"
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>
#include <SofaExplicitOdeSolver/RungeKutta4Solver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include "sofa/core/objectmodel/SPtr.h"

namespace sofa
{

namespace filemanager
{

namespace pml
{


using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;
using namespace sofa::core::objectmodel;

PMLBody::PMLBody()
{
    collisionsON = false;

    AtomsToDOFsIndexes.clear();
}

PMLBody::~PMLBody()
{
}

void PMLBody::createSolver()
{
    if(odeSolverName == "Static") odeSolver = New<StaticSolver>();
    else if(odeSolverName == "EulerImplicit") odeSolver = New<EulerImplicitSolver>();
    else if(odeSolverName == "RungeKutta4") odeSolver = New<RungeKutta4Solver>();
    else if(odeSolverName == "None") return;
    else odeSolver = New<EulerSolver>();

    linearSolver = New<CGLinearSolver< FullMatrix<double>, FullVector<double> > >();

    parentNode->addObject(linearSolver);
    parentNode->addObject(odeSolver);
}


}
}
}
