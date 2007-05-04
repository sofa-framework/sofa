/***************************************************************************
								  PMLBody
                             -------------------
    begin             : August 17th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/02/25 13:51:44 $
    Version           : $Revision: 0.2 $
 ***************************************************************************/

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
using namespace sofa::component::odesolver;

namespace sofa
{

namespace filemanager
{

namespace pml
{


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
