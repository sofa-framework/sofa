#include <sofa/core/fwd.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::core
{

namespace execparams
{
ExecParams* defaultInstance()
{
    return ExecParams::defaultInstance();
}
}

namespace constraintparams
{
const ConstraintParams* defaultInstance()
{
    return ConstraintParams::defaultInstance();
}

ExecParams* dynamicCastToExecParams(sofa::core::ConstraintParams* cparams){ return cparams; }
const ExecParams* dynamicCastToExecParams(const sofa::core::ConstraintParams* cparams){ return cparams; }

}





namespace mechanicalparams
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

ExecParams* dynamicCastToExecParams(sofa::core::MechanicalParams* mparams){ return mparams; }
const ExecParams* dynamicCastToExecParams(const sofa::core::MechanicalParams* mparams){ return mparams; }
}
}

namespace sofa::core::visual
{


namespace visualparams
{
VisualParams* defaultInstance(){ return VisualParams::defaultInstance(); }

sofa::core::ExecParams* dynamicCastToExecParams(sofa::core::visual::VisualParams* vparams){return vparams;}
const sofa::core::ExecParams* dynamicCastToExecParams(const sofa::core::visual::VisualParams* vparams){return vparams;}

sofa::core::visual::DrawTool* getDrawTool(VisualParams* params){ return params->drawTool(); }
sofa::core::visual::DisplayFlags& getDisplayFlags(VisualParams* params){ return params->displayFlags(); }
sofa::core::visual::DrawTool* getDrawTool(const VisualParams* params){ return params->drawTool(); }
const sofa::core::visual::DisplayFlags& getDisplayFlags(const VisualParams* params){ return params->displayFlags(); }
}


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

