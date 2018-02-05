/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/config.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaLoader/initLoader.h>
#include <SofaEngine/initEngine.h>

#include <SofaRigid/initRigid.h>
#include <SofaDeformable/initDeformable.h>
#include <SofaSimpleFem/initSimpleFEM.h>
#include <SofaObjectInteraction/initObjectInteraction.h>
#include <SofaMeshCollision/initMeshCollision.h>
#include <SofaExplicitOdeSolver/initExplicitODESolver.h>
#include <SofaImplicitOdeSolver/initImplicitODESolver.h>
#include <SofaEigen2Solver/initEigen2Solver.h>
#include <SofaObjectInteraction/initObjectInteraction.h>

namespace sofa
{

namespace component
{


void initComponentCommon()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }

    initLoader();
    initEngine();
    initRigid();
    initDeformable();
    initSimpleFEM();
    initObjectInteraction();
    initMeshCollision();
    initExplicitODESolver();
    initImplicitODESolver();
    initEigen2Solver();
    initObjectInteraction();
}

} // namespace component

} // namespace sofa
