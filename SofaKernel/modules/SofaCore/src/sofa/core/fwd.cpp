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
#include <sofa/core/fwd.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/CollisionModel.h>

#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraint.h>

#include <sofa/core/collision/Pipeline.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/ContextObject.h>

#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopology.h>

#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/Shader.h>

namespace sofa::core::execparams
{
ExecParams* defaultInstance()
{
    return ExecParams::defaultInstance();
}
}

namespace sofa::core::constraintparams
{
const ConstraintParams* defaultInstance()
{
    return ConstraintParams::defaultInstance();
}

ExecParams* castToExecParams(sofa::core::ConstraintParams* cparams){ return cparams; }
const ExecParams* castToExecParams(const sofa::core::ConstraintParams* cparams){ return cparams; }

}


namespace sofa::core::mechanicalparams
{

const MechanicalParams* defaultInstance()
{
    return MechanicalParams::defaultInstance();
}

SReal kFactorIncludingRayleighDamping(const sofa::core::MechanicalParams* mparams, SReal d)
{
    return mparams->kFactorIncludingRayleighDamping(d);
}
SReal mFactorIncludingRayleighDamping(const sofa::core::MechanicalParams* mparams, SReal d)
{
    return mparams->mFactorIncludingRayleighDamping(d);
}
SReal dt(const sofa::core::MechanicalParams* mparams)
{
    return mparams->dt();
}
SReal bFactor(const sofa::core::MechanicalParams* mparams)
{
    return mparams->bFactor();
}
SReal kFactor(const sofa::core::MechanicalParams* mparams)
{
    return mparams->kFactor();
}

ExecParams* castToExecParams(sofa::core::MechanicalParams* mparams){ return mparams; }
const ExecParams* castToExecParams(const sofa::core::MechanicalParams* mparams){ return mparams; }
}

namespace sofa::core::visual::visualparams
{
VisualParams* defaultInstance(){ return VisualParams::defaultInstance(); }

sofa::core::ExecParams* castToExecParams(sofa::core::visual::VisualParams* vparams){return vparams;}
const sofa::core::ExecParams* castToExecParams(const sofa::core::visual::VisualParams* vparams){return vparams;}

sofa::core::visual::DrawTool* getDrawTool(VisualParams* params){ return params->drawTool(); }
sofa::core::visual::DisplayFlags& getDisplayFlags(VisualParams* params){ return params->displayFlags(); }
sofa::core::visual::DrawTool* getDrawTool(const VisualParams* params){ return params->drawTool(); }
const sofa::core::visual::DisplayFlags& getDisplayFlags(const VisualParams* params){ return params->displayFlags(); }
}

namespace sofa::core::topology
{

std::ostream& operator<< ( std::ostream& out, const TopologyChange* t )
{
    if (t)
    {
        t->write(out);
    }
    return out;
}

/// Input (empty) stream
std::istream& operator>> ( std::istream& in, TopologyChange*& t )
{
    if (t)
    {
        t->read(in);
    }
    return in;
}

/// Input (empty) stream
std::istream& operator>> ( std::istream& in, const TopologyChange*& )
{
    return in;
}

}

namespace sofa::core
{

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BaseState);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BaseMapping);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BehaviorModel);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::CollisionModel);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::BaseObject);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::ContextObject);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::ConfigurationSetting);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseAnimationLoop);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseMass);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::OdeSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::ConstraintSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseLinearSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseMechanicalState);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseForceField);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseInteractionForceField);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseProjectiveConstraintSet);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseConstraintSet);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::Topology);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::BaseMeshTopology);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::BaseTopologyObject);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::collision::Pipeline);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualLoop);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::Shader);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualModel);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualManager);

}

namespace sofa::core::objectmodel::basecontext
{

SReal getDt(sofa::core::objectmodel::BaseContext* context)
{
    return context->getDt();
}

SReal getTime(sofa::core::objectmodel::BaseContext* context)
{
    return context->getTime();
}

}

