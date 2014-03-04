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
#include <ComponentGeneral/initComponentGeneral.h>
#include <Validation/initValidation.h>
#include <Exporter/initExporter.h>
#include <Engine/initEngine.h>
#include <GraphComponent/initGraphComponent.h>
#include <TopologyMapping/initTopologyMapping.h>
#include <BoundaryCondition/initBoundaryCondition.h>
#include <UserInteraction/initUserInteraction.h>
#include <Constraint/initConstraint.h>
#include <Haptics/initHaptics.h>
#include <DenseSolver/initDenseSolver.h>
#ifdef SOFA_HAVE_CSPARSE
#include <SparseSolver/initSparseSolver.h>
#endif
#ifdef SOFA_HAVE_TAUCS
#include <TaucsSolver/initTaucsSolver.h>
#endif
#ifdef SOFA_HAVE_PARDISO
#include <sofa/component/initPardisoSolver.h>
#endif

#include <Preconditioner/initPreconditioner.h>
#include <OpenglVisual/initOpenGLVisual.h>


namespace sofa
{

namespace component
{


void initComponentGeneral()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }

    initValidation();
    initExporter();
    initEngine();
    initGraphComponent();
    initTopologyMapping();
    initBoundaryCondition();
    initUserInteraction();
    initConstraint();
    initHaptics();
    initDenseSolver();
#ifdef SOFA_HAVE_CSPARSE
    initSparseSolver();
#endif
#ifdef SOFA_HAVE_TAUCS
    initTaucsSolver();
#endif
#ifdef SOFA_HAVE_PARDISO
    initPardisoSolver();
#endif

    initPreconditioner();
#ifndef SOFA_NO_OPENGL
    initOpenGLVisual();
#endif
}


} // namespace component

} // namespace sofa
