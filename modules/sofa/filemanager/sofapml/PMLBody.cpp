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

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PMLBody.h"
#include "sofa/component/odesolver/CGImplicitSolver.h"
#include "sofa/component/odesolver/EulerSolver.h"
#include "sofa/component/odesolver/StaticSolver.h"
#include "sofa/component/odesolver/RungeKutta4Solver.h"

namespace sofa
{

namespace filemanager
{

namespace pml
{


using namespace sofa::component::odesolver;

PMLBody::PMLBody()
{
    collisionsON = false;

    mass=NULL;
    topology=NULL;
    forcefield=NULL;
    mmodel=NULL;
    solver = NULL;

    AtomsToDOFsIndexes.clear();
}

PMLBody::~PMLBody()
{
    if(mass) delete mass;
    if(topology) delete topology;
    if(forcefield) delete forcefield;
    if (solver) delete solver;
}

void PMLBody::createSolver()
{
    if(solverName == "Static") solver = new StaticSolver;
    else if(solverName == "Euler") solver = new EulerSolver;
    else if(solverName == "RungeKutta4") solver = new RungeKutta4Solver;
    else if(solverName == "None") return;
    else solver = new CGImplicitSolver;

    parentNode->addObject(solver);
}


}
}
}
