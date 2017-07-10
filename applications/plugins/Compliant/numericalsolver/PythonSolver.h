#ifndef COMPLIANT_NUMERICALSOLVER_PYTHONSOLVER_H
#define COMPLIANT_NUMERICALSOLVER_PYTHONSOLVER_H

#include <Compliant/numericalsolver/KKTSolver.h>
#include <Compliant/python/python.h>

namespace sofa {
namespace component {
namespace linearsolver {

class SOFA_Compliant_API PythonSolver : public KKTSolver {


    struct block {
        std::size_t offset, size;
        Constraint* projector;
    };
    
    std::vector<block> blocks;
    
    struct data_type {
        const system_type* sys;
        const python::vec<block>* blocks;
        void(*project)(real* vec, const block* b);
    };
    
    typedef void(*factor_callback_type)(const data_type* data);
    typedef void(*solve_callback_type)(vec* res, const data_type* data, const vec* rhs);
    typedef void(*correct_callback_type)(vec* res, const data_type* data, const vec* rhs, double damping);    
    
    // TODO correction damping
    Data< python::opaque< factor_callback_type > > factor_callback;
    Data< python::opaque< solve_callback_type > > solve_callback;
    Data< python::opaque< correct_callback_type > > correct_callback;    

protected:
    static void fetch_blocks(std::vector<block>& res, const system_type& sys);
    static void project(real* vec, const block* b);
    
public:
    
    PythonSolver();
    
    virtual void factor(const system_type& system);

    virtual void solve(vec& x,
                       const system_type& system,
                       const vec& rhs) const;

    virtual void correct(vec& x,
                         const system_type& system,
                         const vec& rhs,
                         real damping) const;

    
};
    
}
}
}


#endif
