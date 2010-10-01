#ifndef SOFA_SIMULATION_COMMON_VECTOROPERATIONS_H
#define SOFA_SIMULATION_COMMON_VECTOROPERATIONS_H

#include <sofa/core/behavior/BaseVectorOperations.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>


namespace sofa
{
namespace core
{
class ExecParams;
}
namespace simulation
{
namespace common
{

class VectorOperations : public sofa::core::behavior::BaseVectorOperations
{

public:

    VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::Context* ctx);

    /// Allocate a temporary vector
    void v_alloc(sofa::core::MultiVecCoordId& v);
    void v_alloc(sofa::core::MultiVecDerivId& v);
    /// Free a previously allocated temporary vector
    void v_free(sofa::core::MultiVecCoordId& id);
    void v_free(sofa::core::MultiVecDerivId& id);

    void v_clear(core::MultiVecId v); ///< v=0
    void v_eq(core::MultiVecId v, core::MultiVecId a); ///< v=a
    void v_peq(core::MultiVecId v, core::MultiVecId a, double f=1.0); ///< v+=f*a
#ifdef SOFA_SMP
    void v_peq(core::MultiVecId v, core::VecId a, Shared<double> &fSh, double f=1.0) ; ///< v+=f*a
    void v_meq(core::MultiVecIdv, core::VecId a, Shared<double> &fSh) ; ///< v+=f*a
#endif
    void v_teq(core::MultiVecId v, double f) ; ///< v*=f
    void v_op(core::MultiVecId v, core::MultiVecId a, core::MultiVecId b, double f=1.0) ; ///< v=a+b*f
#ifdef SOFA_SMP
    void v_op(core::MultiVecId v, core::MultiVecId a, core::MultiVecId b, Shared<double> &f) ; ///< v=a+b*f
#endif
    void v_dot(core::MultiVecId a, core::MultiVecId b) = 0; ///< a dot b ( get result using finish )
#ifdef SOFA_SMP
    void v_dot(Shared<double> &result,core::MultiVecId a, core::MultiVecId b) ; ///< a dot b
#endif
    void v_threshold(core::MultiVecId a, double threshold); ///< nullify the values below the given threshold

    void print( core::MultiVecId v, std::ostream& out );
protected:

    void prepareVisitor(Visitor* v);
    void prepareVisitor(MechanicalVisitor* v);

    template<class T>
    void executeVisitor(T v, bool prefetch = false)
    {
        prepareVisitor(&v);
        v.execute( ctx, prefetch );
    }

    //template<class T>
    //void executeVisitor(T* v)
    //{
    //    prepareVisitor(v);
    //    v->execute( this->getContext() );
    //}

    //template<class T>
    //void executeVisitor(T v, bool prefetch)
    //{
    //    prepareVisitor(&v);
    //    v.execute( this->getContext(), prefetch );
    //}

    //template<class T>
    //void executeVisitor(T* v, bool prefetch)
    //{
    //    prepareVisitor(v);
    //    v->execute( this->getContext(), prefetch );
    //}

    /// Result of latest v_dot operation
    double result;
};

}
}
}


#endif //SOFA_SIMULATION_COMMON_VECTOROPERATIONS_H
