#ifndef SOFA_SIMULATION_COMMON_VISITOREXECUTE_H
#define SOFA_SIMULATION_COMMON_VISITOREXECUTE_H


#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

namespace sofa
{
namespace simulation
{
namespace common
{

struct VisitorExecuteFunc
{
protected:
    core::objectmodel::BaseContext& ctx;
public:
    VisitorExecuteFunc(core::objectmodel::BaseContext& ctx):ctx(ctx) {};

    template< class Visitor >
    void operator()(Visitor* pv)
    {
        prepareVisitor(pv);
        pv->execute(&ctx);
    }
    template< class Visitor >
    void operator()(Visitor v)
    {
        prepareVisitor(&v);
        v.execute(&ctx);
    }
protected:
    void prepareVisitor( sofa::simulation::Visitor* v)
    {
        v->setTags( ctx.getTags() );
    }
    void prepareVisitor( sofa::simulation::BaseMechanicalVisitor* mv)
    {
        prepareVisitor( (sofa::simulation::Visitor*)mv );
    }
};
}
}
}

#endif // SOFA_SIMULATION_COMMON_VISITOREXECUTE_H
