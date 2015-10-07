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
#include <sofa/helper/system/config.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaLoader/initLoader.h>
/*
#include <SofaRigid/initRigid.h>
#include <SofaDeformable/initDeformable.h>
#include <SofaSimpleFem/initSimpleFEM.h>
#include <SofaObjectInteraction/initObjectInteraction.h>
#include <SofaMeshCollision/initMeshCollision.h>
#include <SofaExplicitOdeSolver/initExplicitODESolver.h>
#include <SofaImplicitOdeSolver/initImplicitODESolver.h>
*/

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
/*
    initRigid();
    initDeformable();
    initSimpleFEM();
    initObjectInteraction();
    initMeshCollision();
    initExplicitODESolver();
    initImplicitODESolver();
*/
}

} // namespace component

} // namespace sofa
