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
#include <sofa/helper/system/config.h>

#include <sofa/component/behaviormodel/initBehaviorModel.h>
#include <sofa/component/collision/initCollision.h>
#include <sofa/component/configurationsetting/initConfigurationSetting.h>
#include <sofa/component/projectiveconstraintset/initProjectiveConstraintSet.h>
#include <sofa/component/constraintset/initConstraintSet.h>
#include <sofa/component/container/initContainer.h>
#include <sofa/component/contextobject/initContextObject.h>
#include <sofa/component/controller/initController.h>
#include <sofa/component/engine/initEngine.h>
#ifdef SOFA_DEV
#include <sofa/component/fem/quadrature/initQuadrature.h>
#include <sofa/component/fem/fetype/initFEType.h>
#include <sofa/component/fem/straintensor/initStrainTensor.h>
#include <sofa/component/fem/material/initMaterial.h>
#include <sofa/component/fem/initFEM.h>
#include <sofa/component/fem/forcefield/initFEMForceField.h>
#endif
#include <sofa/component/forcefield/initForceField.h>
#include <sofa/component/interactionforcefield/initInteractionForceField.h>
#include <sofa/component/linearsolver/initLinearSolver.h>
#include <sofa/component/mapping/initMapping.h>
#include <sofa/component/mass/initMass.h>
#include <sofa/component/mastersolver/initMasterSolver.h>
#include <sofa/component/misc/initMisc.h>
#include <sofa/component/odesolver/initOdeSolver.h>
#include <sofa/component/topology/initTopology.h>
#include <sofa/component/visualmodel/initVisualModel.h>
#include <sofa/component/init.h>
#include <sofa/component/loader/initLoader.h>

namespace sofa
{

namespace component
{


void init()
{
    static bool first = true;
    if (first)
    {
        initBehaviorModel();
        initCollision();
        initConfigurationSetting();
        initProjectiveConstraintSet();
        initConstraintSet();
        initContainer();
        initContextObject();
        initController();
        initEngine();
#ifdef SOFA_DEV
        initQuadrature();
        initFEType();
        initStrainTensor();
        initMaterial();
        initFEM();
        initFEMForceField();
#endif
        initForceField();
        initInteractionForceField();
        initLinearSolver();
        initMapping();
        initMass();
        initMasterSolver();
        initMisc();
        initOdeSolver();
        initTopology();
        initVisualModel();
        initLoader();

        first = false;
    }
}

} // namespace component

} // namespace sofa
