#ifndef SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H
#define SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/objectmodel/Context.h>


namespace sofa
{
namespace core
{
namespace behavior
{

class BaseVectorOperation
{
public:
    BaseVectorOperation(core::objectmodel::Context*, core::ExecParams* );

    /// Allocate a temporary vector
    virtual core::VecId v_alloc(core::VecType t) = 0;
    /// Free a previously allocated temporary vector
    virtual void v_free(core::MultiVecId v) = 0;

    virtual void v_clear(core::MultiVecId v) = 0; ///< v=0
    virtual void v_eq(core::MultiVecId v, core::MultiVecId a) = 0; ///< v=a
    virtual void v_peq(core::MultiVecId v, core::MultiVecId a, double f=1.0) = 0; ///< v+=f*a
#ifdef SOFA_SMP
    virtual void v_peq(core::VecId v, core::VecId a, Shared<double> &fSh, double f=1.0) = 0; ///< v+=f*a
    virtual void v_meq(core::VecId v, core::VecId a, Shared<double> &fSh) = 0; ///< v+=f*a
#endif
    virtual void v_teq(core::VecId v, double f) = 0; ///< v*=f
    virtual void v_op(core::VecId v, core::MultiVecId a, core::MultiVecId b, double f=1.0) = 0; ///< v=a+b*f
#ifdef SOFA_SMP
    virtual void v_op(core::MultiVecId v, core::MultiVecId a, core::MultiVecId b, Shared<double> &f) = 0; ///< v=a+b*f
#endif
    virtual void v_dot(core::MultiVecId a, core::MultiVecId b) = 0; ///< a dot b ( get result using finish )
#ifdef SOFA_SMP
    virtual void v_dot(Shared<double> &result,core::MultiVecId a, core::MultiVecId b) = 0; ///< a dot b
#endif
    virtual void v_threshold(core::MultiVecId a, double threshold) = 0; ///< nullify the values below the given threshold

    virtual void print( core::MultiVecId v, std::ostream& out );

};

}
}
}

#endif //SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H
