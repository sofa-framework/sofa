/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_H
#define SOFA_COMPONENT_H

#ifdef WIN32
	#pragma message ( "component.h : This file is useless. If you need it, you are doing something wrong." )
#else
	#warning component.h This file is useless. If you need it, you are doing something wrong.
#endif

#include <sofa/helper/system/config.h>

#include <SofaBaseAnimationLoop/config.h>
#include <SofaBaseCollision/config.h>
#include <SofaBaseLinearSolver/config.h>
#include <SofaBaseMechanics/config.h>
#include <SofaBaseTopology/config.h>
#include <SofaBaseVisual/config.h>
#include <SofaBoundaryCondition/config.h>
#include <SofaComponentAdvanced/config.h>
#include <SofaComponentBase/config.h>
#include <SofaComponentCommon/config.h>
#include <SofaComponentGeneral/config.h>
#include <SofaComponentMain/config.h>
#include <SofaComponentMisc/config.h>
#include <SofaConstraint/config.h>
#include <SofaDeformable/config.h>
#include <SofaDenseSolver/config.h>
#include <SofaEigen2Solver/config.h>
#include <SofaEngine/config.h>
#include <SofaEulerianFluid/config.h>
#include <SofaExplicitOdeSolver/config.h>
#include <SofaExporter/config.h>
#include <SofaGraphComponent/config.h>
#include <SofaHaptics/config.h>
#include <SofaImplicitOdeSolver/config.h>
#include <SofaLoader/config.h>
#include <SofaMeshCollision/config.h>
#include <SofaMisc/config.h>
#include <SofaMiscCollision/config.h>
#include <SofaMiscEngine/config.h>
#include <SofaMiscFem/config.h>
#include <SofaMiscForceField/config.h>
#include <SofaMiscMapping/config.h>
#include <SofaMiscSolver/config.h>
#include <SofaMiscTopology/config.h>
#include <SofaNonUniformFem/config.h>
#include <SofaObjectInteraction/config.h>
#include <SofaOpenglVisual/config.h>
#include <SofaPardisoSolver/config.h>
#include <SofaPreconditioner/config.h>
#include <SofaRigid/config.h>
#include <SofaSimpleFem/config.h>
#include <SofaSparseSolver/config.h>
#include <SofaSphFluid/config.h>
#include <SofaTaucsSolver/config.h>
#include <SofaTopologyMapping/config.h>
#include <SofaUserInteraction/config.h>
#include <SofaValidation/config.h>
#include <SofaVolumetricData/config.h>

#endif //SOFA_COMPONENT_H
