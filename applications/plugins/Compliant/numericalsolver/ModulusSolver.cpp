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
static int ModulusSolverClass = core::RegisterObject("Modulus solver").add< ModulusSolver >();




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
    
    unsigned off = 0;
    for(unsigned i = 0, n = sys.constraints.size(); i < n; ++i) {
        const unsigned dim = sys.compliant[i]->getMatrixSize();

        if( sys.constraints[i].projector ) {

            const std::type_info& type = typeid(*sys.constraints[i].projector);
            
            if( type == typeid(CoulombConstraint) ) {
                unilateral( off ) = 1;
            } else if ( type == typeid(UnilateralConstraint) ) {
                unilateral.segment(off, dim).setOnes();
            } 
        }
        
        off += dim;
    }

    // build system TODO hardcoded regularization
    sub.projected_kkt(sys, 1e-14, true);
    
    const vec Hdiag_inv = sys.P * sys.H.diagonal().cwiseInverse();

    diagonal = vec::Zero(sys.n);
    
    // change diagonal compliance for unilateral constraints
    const real omega = this->omega.getValue();
    for(unsigned i = 0; i < sys.n; ++i) {

        if( unilateral(i) ) {

            diagonal(i) = sys.J.row(i).dot( Hdiag_inv.asDiagonal() * sys.J.row(i).transpose());
            // diagonal(i) = 1;

            sub.matrix.coeffRef(sub.primal.cols() + i,
                                sub.primal.cols() + i) = -omega * diagonal(i);
        }
    }

    // std::cout << unilateral.transpose() << std::endl;
    // std::cout << "Hinv " << Hdiag_inv.transpose() << std::endl;
    // std::cout << "diagonal " << diagonal.transpose() << std::endl;

    // factor the modified kkt
    sub.factor( solver );
}


// good luck with that :p
void ModulusSolver::solve(vec& res,
                          const system_type& sys,
                          const vec& rhs) const {

    vec constant = rhs;
    if( sys.n ) constant.tail(sys.n) = -constant.tail(sys.n);

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
    
    const real omega = this->omega.getValue();
    const real precision = this->precision.getValue();

    utils::anderson accel(sys.n, anderson.getValue(), diagonal);
    
    unsigned k;
    const unsigned kmax = iterations.getValue();
    for(k = 0; k < kmax; ++k) {
        
        // tmp = (x, |z|)
        tmp = y;
        
        for( unsigned i = 0, imax = sys.n; i < imax; ++i) {
            if( unilateral(i) ) {
                tmp(sys.m + i) = std::abs(tmp(sys.m + i));
            }
        }

        // store |z|
        zabs = unilateral.cwiseProduct(tmp.tail(sys.n));
        
        // prod is wrong, we need to add back 2 diag * omega |z|
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
        y.tail(sys.n).array() -= tmp.tail(sys.n).array() + unilateral.array() * y.tail(sys.n).array();

        if( anderson.getValue() ) {

            // reuse zabs
            zabs = y.tail(sys.n);
            accel(zabs, true);
            y.tail(sys.n) = zabs;

        }
        
        error = unilateral.cwiseProduct(old - y.tail(sys.n)).norm();
        if( error <= precision ) break;
    }

    sout << "fixed point error: " << error << "\titerations: " << k << sendl;

    // and that should be it ?
    res = y;

    // add |z| to dual to get lambda
    res.tail(sys.n).array() += unilateral.array() * y.tail(sys.n).array().abs();

    // TODO we should recompute x based on actual, non sticky lambdas
    
}


}
}
}

