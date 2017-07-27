#include "ConstraintStoreLambdaVisitor.h"

namespace sofa
{
namespace simulation
{

ConstraintStoreLambdaVisitor::ConstraintStoreLambdaVisitor(const sofa::core::ConstraintParams* cParams, const sofa::defaulttype::BaseVector* lambda)
:BaseMechanicalVisitor(cParams)
,m_cParams(cParams)
,m_lambda(lambda)
{
}

Visitor::Result ConstraintStoreLambdaVisitor::fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
{
    if (core::behavior::BaseConstraint *c = core::behavior::BaseConstraint::DynamicCast(cSet))
    {
        ctime_t t0 = begin(node, c);
        c->storeLambda(m_cParams, m_cParams->lambda(), m_lambda);
        end(node, c, t0);
    }
    return RESULT_CONTINUE;
}

void ConstraintStoreLambdaVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    sofa::core::MechanicalParams mparams(*m_cParams);
    mparams.setDx(m_cParams->dx());
    mparams.setF(m_cParams->lambda());
    map->applyJT(&mparams, m_cParams->lambda(), m_cParams->lambda());
}

}

}
