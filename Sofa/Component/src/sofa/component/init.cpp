/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/component/init.h>

#include <sofa/component/animationloop/init.h>
#include <sofa/component/collision/init.h>
#include <sofa/component/constraint/init.h>
#include <sofa/component/controller/init.h>
#include <sofa/component/diffusion/init.h>
#include <sofa/component/engine/init.h>
#include <sofa/component/haptics/init.h>
#include <sofa/component/io/init.h>
#include <sofa/component/linearsolver/init.h>
#include <sofa/component/linearsystem/init.h>
#include <sofa/component/mapping/init.h>
#include <sofa/component/mass/init.h>
#include <sofa/component/mechanicalload/init.h>
#include <sofa/component/odesolver/init.h>
#include <sofa/component/playback/init.h>
#include <sofa/component/sceneutility/init.h>
#include <sofa/component/setting/init.h>
#include <sofa/component/solidmechanics/init.h>
#include <sofa/component/statecontainer/init.h>
#include <sofa/component/topology/init.h>
#include <sofa/component/visual/init.h>

namespace sofa::component
{
    
extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

void init()
{
    static bool first = true;
    if (first)
    {
        sofa::component::animationloop::init();
        sofa::component::collision::init();
        sofa::component::constraint::init();
        sofa::component::controller::init();
        sofa::component::diffusion::init();
        sofa::component::engine::init();
        sofa::component::haptics::init();
        sofa::component::io::init();
        sofa::component::linearsystem::init();
        sofa::component::linearsolver::init();
        sofa::component::mapping::init();
        sofa::component::mass::init();
        sofa::component::mechanicalload::init();
        sofa::component::odesolver::init();
        sofa::component::playback::init();
        sofa::component::sceneutility::init();
        sofa::component::setting::init();
        sofa::component::solidmechanics::init();
        sofa::component::statecontainer::init();
        sofa::component::topology::init();
        sofa::component::visual::init();

        first = false;
    }
}

} // namespace sofa::component
