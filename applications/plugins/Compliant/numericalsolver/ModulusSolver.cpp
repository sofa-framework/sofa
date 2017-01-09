#include "ModulusSolver.h"
#include "EigenSparseResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/anderson.h"
#include "../utils/nlnscg.h"

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
int ModulusSolverClass = core::RegisterObject("Modulus solver").add< ModulusSolver >();




ModulusSolver::ModulusSolver() 
    : omega(initData(&omega,
                     (real)1.0,
                     "omega",
                     "magic stuff")),
      anderson(initData(&anderson, unsigned(0),
                        "anderson",
                        "anderson acceleration history size, 0 if none")),
      nlnscg(initData(&nlnscg, false,
                        "nlnscg",
                        "nlnscg acceleration"))
      
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

    // compute diagonal + homogenize tangent values
    {
        unsigned off = 0;
    
        for(unsigned i = 0, n = sys.constraints.size(); i < n; ++i) {
            const unsigned dim = sys.compliant[i]->getMatrixSize();

            for(unsigned j = off, je = off + dim; j < je; ++j) {
                diagonal(j) = sys.J.row(j).dot( Hdiag_inv.asDiagonal() * sys.J.row(j).transpose());
            }

            typedef linearsolver::CoulombConstraintBase proj_type;
            Constraint* c = sys.constraints[i].projector.get();

            // unilateral mask
            if( c ) {
                unilateral.segment(off, dim).setOnes();
            }
            
            if( c && proj_type::checkConstraintType(c) ) {
                assert( dim % 3 == 0 && "non vec3 dofs not handled");
                for(unsigned j = 0, je = dim / 3; j < je; ++j) {
                    
                    // set both tangent values to max tangent value to
                    // avoid friction anisotropy
                    
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
    }

    // update system matrix (2, 2 block) with diagonal
    for(unsigned i = 0; i < sys.n; ++i) {

        sub.matrix.coeffRef(sub.primal.cols() + i,
                            sub.primal.cols() + i) += unilateral(i) * (-omega * diagonal(i));
        
    }

    // factor the modified kkt
    sub.factor( solver );

    // std::clog << unilateral.transpose() << std::endl;
}


template<class View>
static void project(View&& view, const ModulusSolver::system_type& sys, bool correct) {
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

    if( !sys.n ) {
        sub.solve(solver, res, rhs);
        return;
    }
    
    vec constant, tmp;
    constant.resize(sys.size() );
    tmp.resize(sys.size() );    

    constant.head(sys.m) = rhs.head(sys.m);
    constant.tail(sys.n) = - rhs.tail(sys.n);
    
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

    // // utils::anderson accel(sys.n, anderson.getValue(), diagonal);
    // utils::nlnscg accel(sys.n, diagonal);    
    
    unsigned k;
    const unsigned kmax = iterations.getValue();

    const bool use_accel = anderson.getValue();
    
    for(k = 0; k < kmax; ++k) {
        old = y.tail(sys.n);
        
        // |z| = 2 * P(z) - z
        zabs = y.tail(sys.n);
        project( zabs, sys, correct );
        zabs = 2 * zabs - y.tail(sys.n);

        // don't consider bilateral constraints
        zabs = zabs.array() * unilateral.array();
        
        tmp = constant;

        // TODO only non-projection constraints?
        tmp.head(sys.m) += sys.J.transpose() * zabs;
        tmp.tail(sys.n) += sys.C * zabs - omega * diagonal.cwiseProduct(zabs);
        
        // solve in place
        sub.solve(solver, y, tmp);
        
        error = (old - y.tail(sys.n)).norm();
        if( error <= precision ) break;
    }

    if( this->f_printLog.getValue() ) {
        serr << "fixed point error: " << error << "\titerations: " << k << sendl;
    }

    // and that should be it ?
    res.head(sys.m) = y.head(sys.m);
    res.tail(sys.n) = (y.tail(sys.n) + zabs);
    
}


}
}
}

