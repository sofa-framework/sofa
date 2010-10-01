#ifndef SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H
#define SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H


#include <sofa/core/MultiVecId.h>

namespace sofa
{
namespace core
{

class ExecParams;

namespace objectmodel
{
class Context;
}
namespace behavior
{

class BaseVectorOperations
{

protected:
    const core::ExecParams* params;
    const core::objectmodel::Context* ctx;
    double result;
public:
    BaseVectorOperations(const core::ExecParams* params, const core::objectmodel::Context* ctx):params(params),ctx(ctx) {};

    /// Allocate a temporary vector
    virtual void v_alloc(sofa::core::MultiVecCoordId& id) = 0;
    virtual void v_alloc(sofa::core::MultiVecDerivId& id) = 0;
    /// Free a previously allocated temporary vector
    virtual void v_free(sofa::core::MultiVecCoordId& id) = 0;
    virtual void v_free(sofa::core::MultiVecDerivId& id) = 0;

    virtual void v_clear(core::MultiVecId v) = 0; ///< v=0
    virtual void v_eq(core::MultiVecId v, core::MultiVecId a) = 0; ///< v=a
    virtual void v_peq(core::MultiVecId v, core::MultiVecId a, double f=1.0) = 0; ///< v+=f*a
#ifdef SOFA_SMP
    virtual void v_peq(core::MultiVecId v, core::MultiVecId a, Shared<double> &fSh, double f=1.0) = 0; ///< v+=f*a
    virtual void v_meq(core::MultiVecId v, core::MultiVecId a, Shared<double> &fSh) = 0; ///< v+=f*a
#endif
    virtual void v_teq(core::MultiVecId v, double f) = 0; ///< v*=f
    virtual void v_op(core::MultiVecId v, core::ConstMultiVecId a, core::ConstMultiVecId b, double f=1.0) = 0; ///< v=a+b*f
#ifdef SOFA_SMP
    virtual void v_op(core::MultiVecId v, core::MultiVecId a, core::MultiVecId b, Shared<double> &f) = 0; ///< v=a+b*f
#endif
    virtual void v_dot(core::ConstMultiVecId a, core::ConstMultiVecId b) = 0; ///< a dot b ( get result using finish )
#ifdef SOFA_SMP
    virtual void v_dot(Shared<double> &result,core::MultiVecId a, core::MultiVecId b) = 0; ///< a dot b
#endif
    virtual void v_threshold(core::MultiVecId a, double threshold) = 0; ///< nullify the values below the given threshold


    virtual double finish() = 0;

    virtual void print( core::MultiVecId v, std::ostream& out ) = 0;

};

}
}
}

#endif //SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H

