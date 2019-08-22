#ifndef COMPLIANT_UTILS_SUBKKT_INL
#define COMPLIANT_UTILS_SUBKKT_INL

#include "sub_kkt.h"

namespace utils {


// defaults to response API
template<class Solver>
struct sub_kkt::traits {

    static void factor(Solver& solver, const rmat& matrix) {
        solver.factor(matrix);
    }

    static void solve(const Solver& solver, const vec& res, const vec& rhs) {
        solver.solve(res, rhs);
    }

};


template<class Solver>
void sub_kkt::factor(Solver& solver) const {
    traits<Solver>::factor(solver, matrix);
}


template<class Action>
void sub_kkt::project_unproject(const Action& action, vec& res, const vec& rhs) const {
    
    assert( rhs.size() == size_full() );
    
    // project
    vtmp1.resize( size_sub() );

    if( primal.cols() ) {
        vtmp1.head(primal.cols()).noalias() = primal.transpose() * rhs.head(primal.rows());
    }

    if( dual.cols() ) {
        vtmp1.tail(dual.cols()).noalias() = dual.transpose() * rhs.tail(dual.rows());
    }

    // do stuff (note: no alias)
    vtmp2.resize( size_sub() );
    action(vtmp2, vtmp1);
    
    // unproject
    res.resize( rhs.size() );
    
    if( primal.cols() ) {
        res.head(primal.rows()).noalias() = primal * vtmp2.head(primal.cols());
    }

    if( dual.cols() ) {
        res.tail(dual.rows()).noalias() = dual * vtmp2.tail(dual.cols());
    }

}


template<class Solver>
struct sub_kkt::solve_action {
    const Solver& solver;
    solve_action(const Solver& solver) : solver(solver) { }

    void operator()(vec& lhs, const vec& rhs) const {
        traits<Solver>::solve(solver, lhs, rhs);
    }
    
};


template<class Solver>
void sub_kkt::solve(const Solver& solver, vec& res, const vec& rhs) const {
    assert( rhs.size() == size_full() );

    project_unproject( solve_action<Solver>(solver), res, rhs);
}




}



#endif


