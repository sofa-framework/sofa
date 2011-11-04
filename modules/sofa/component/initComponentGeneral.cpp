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
#include <sofa/component/initComponentGeneral.h>
#include <sofa/component/initValidation.h>
#include <sofa/component/initExporter.h>
#include <sofa/component/initEngine.h>
#include <sofa/component/initGraphComponent.h>
#include <sofa/component/initTopologyMapping.h>
#include <sofa/component/initBoundaryCondition.h>
#include <sofa/component/initUserInteraction.h>
#include <sofa/component/initConstraint.h>
#include <sofa/component/initHaptics.h>
#include <sofa/component/initDenseSolver.h>
#ifdef SOFA_HAVE_CSPARSE
#include <sofa/component/initSparseSolver.h>
#endif
#include <sofa/component/initPreconditioner.h>
#include <sofa/component/initOpenGLVisual.h>


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
    initPreconditioner();
    initOpenGLVisual();

}


} // namespace component

} // namespace sofa
