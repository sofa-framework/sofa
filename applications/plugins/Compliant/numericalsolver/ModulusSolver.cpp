#include "ModulusSolver.h"
#include "EigenSparseResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/anderson.h"
#include "../utils/scoped.h"

#include "../constraint/CoulombConstraint.h"
#include "../constraint/UnilateralConstraint.h"

#include <Eigen/SparseCholesky>
#include "../utils/sub_kkt.inl"


namespace utils {

template<class Matrix>
struct sub_kkt::traits< Eigen::SimplicialLDLT<Matrix, Eigen::Upper> > {

    typedef Eigen::SimplicialLDLT<Matrix, Eigen::Upper> solver_type;
    
    static void factor(solver_type& solver, const rmat& matrix) {
        // ldlt expects a cmat so we transpose to ease the conversion
        solver.compute(matrix.triangularView<Eigen::Lower>().transpose());
        if( solver.info() != Eigen::Success ) {
            std::cerr << "error during LDLT factorization !" << std::endl;

            std::cerr << matrix << std::endl;
        }
    }

    static void solve(const solver_type& solver, vec& res, const vec& rhs) {
        res = solver.solve(rhs);
    }

};

}


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(ModulusSolver)
const int ModulusSolverClass = core::RegisterObject("Modulus solver").add< ModulusSolver >();




ModulusSolver::ModulusSolver() 
    : omega(initData(&omega,
                     (real)1.0,
                     "omega",
                     "magic stuff")),
      anderson(initData(&anderson, unsigned(0),
                        "anderson",
                        "anderson acceleration history size, 0 if none"))
{
    
}

ModulusSolver::~ModulusSolver() {
    
}


void ModulusSolver::factor(const system_type& sys) {

    // build unilateral mask
    unilateral = vec::Zero(sys.n);
    
    // build system TODO hardcoded regularization
    sub.projected_kkt(sys, 1e-14, true);
    
    const vec Hdiag_inv = sys.P * sys.H.diagonal().cwiseInverse();

    diagonal = vec::Zero(sys.n);
    
    // change diagonal compliance 
    const real omega = this->omega.getValue();

    // homogenize tangent diagonal values
    unsigned off = 0;
    
    for(unsigned i = 0, n = sys.constraints.size(); i < n; ++i) {
        const unsigned dim = sys.compliant[i]->getMatrixSize();

        for(unsigned j = off, je = off + dim; j < je; ++j) {
            diagonal(j) = sys.J.row(j).dot( Hdiag_inv.asDiagonal() * sys.J.row(j).transpose());
        }

        typedef linearsolver::CoulombConstraintBase proj_type;
        Constraint* c = sys.constraints[i].projector.get();
        if( c && proj_type::checkConstraintType(c) ) {
            assert( dim % 3 == 0 && "non vec3 dofs not handled");
            for(unsigned j = 0, je = dim / 3; j < je; ++j) {
                // set both tangent values to max tangent value
                const unsigned t1 = off + 3 * j + 1;
                const unsigned t2 = off + 3 * j + 2;
                
                const SReal value = std::max(diagonal(t1),
                                             diagonal(t2));

                diagonal(t1) = value;
                diagonal(t2) = value;
            }
                    
        }
        
        off += dim;
    }
    assert( off == sys.n );

    // update system matrix with diagonal
    for(unsigned i = 0; i < sys.n; ++i) {

        sub.matrix.coeffRef(sub.primal.cols() + i,
                            sub.primal.cols() + i) = -omega * diagonal(i);
    }

    // factor the modified kkt
    sub.factor( solver );
}

 
void ModulusSolver::project(vec::SegmentReturnType view, const system_type& sys, bool correct) const {
    unsigned off = 0;
    
    for(unsigned i = 0, n = sys.constraints.size(); i < n; ++i) {
        const unsigned dim = sys.compliant[i]->getMatrixSize();
        
        const linearsolver::Constraint* proj = sys.constraints[i].projector.get();
        if( proj ) {
            proj->project(&view[off], dim, 0, correct);
        }
        
        off += dim;
    }
    assert( off == view.size() );
}

// good luck with that :p
void ModulusSolver::solve(vec& res,
                          const system_type& sys,
                          const vec& rhs) const {

    vec constant = rhs;
    if( sys.n ) constant.tail(sys.n) = - constant.tail(sys.n);

    vec tmp;
    sub.solve(solver, tmp, constant);
    
    if(!sys.n) {
        res = tmp;
        return;
    }

    // y = (x, z)
    vec y = vec::Zero( sys.size() );

    // TODO warm start ?
    vec zabs;
    vec old;
    real error;

    // TODO
    const bool correct = false;
    
    const real omega = this->omega.getValue();
    const real precision = this->precision.getValue();

    utils::anderson accel(sys.n, anderson.getValue(), diagonal);
    
    unsigned k;
    const unsigned kmax = iterations.getValue();
    for(k = 0; k < kmax; ++k) {
        
        // tmp = (x, |z|)
        tmp = y;

        // |z| = 2 * P(z) - z
        project( tmp.tail(sys.n), sys, correct );
        tmp.tail(sys.n) = 2 * tmp.tail(sys.n) - y.tail(sys.n);

        zabs = tmp.tail(sys.n);

        // TODO fix this mess
        // after this, prod is wrong, we need to add back 2 diag * omega |z|
        {
            scoped::timer step("subkkt prod");
            sub.prod(tmp, tmp, true);
        }
        
        // tmp += 2 * omega.getValue() * zabs;
        tmp.tail(sys.n).array() += (2 * omega) * diagonal.array() * zabs.array();

        // 
        tmp -= constant;

        // solve
        sub.solve(solver, tmp, tmp);
        
        // backup old dual
        old = y.tail(sys.n);
        
        y.head(sys.m).array() -= tmp.head(sys.m).array();
        y.tail(sys.n).array() -= tmp.tail(sys.n).array() + y.tail(sys.n).array();

        if( anderson.getValue() ) {

            // reuse zabs
            zabs = y.tail(sys.n);
            accel(zabs, true);
            y.tail(sys.n) = zabs;

        }
        
        error = (old - y.tail(sys.n)).norm();
        if( error <= precision ) break;
    }

    sout << "fixed point error: " << error << "\titerations: " << k << sendl;

    // and that should be it ?
    res = y;

    // we need to project 2 z to get lambdas
    res.tail(sys.n) *= 2.0;
    project( res.tail(sys.n), sys, correct );
    
    // TODO we should recompute x based on actual, non sticky lambdas
    
}


}
}
}

