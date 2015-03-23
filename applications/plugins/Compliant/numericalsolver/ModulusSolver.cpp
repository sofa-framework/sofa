#include "ModulusSolver.h"
#include "EigenSparseResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/sparse.h"


#include "../constraint/CoulombConstraint.h"
#include "../constraint/UnilateralConstraint.h"

#include <Eigen/SparseCholesky>
#include "SubKKT.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(ModulusSolver)
static int ModulusSolverClass = core::RegisterObject("Modulus solver").add< ModulusSolver >();

typedef AssembledSystem::vec vec;



ModulusSolver::ModulusSolver() 
    : omega(initData(&omega,
                     1.0,
                     "omega",
                     "magic stuff")) 
{
    
}

ModulusSolver::~ModulusSolver() {
    
}


void ModulusSolver::init() {

    KKTSolver::init();

    // let's find a response
    response = this->getContext()->get<Response>( core::objectmodel::BaseContext::Local );

    // fallback in case we missed
    if( !response ) {
        response = new LDLTResponse();
        this->getContext()->addObject( response );
        
        serr << "fallback response class: "
             << response->getClassName()
             << " added to the scene" << sendl;
    }
}


void ModulusSolver::factor(const AssembledSystem& sys) {

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

    // build system
    SubKKT::projected_kkt(sub, sys, 1e-14 );

    const vec Hdiag_inv = sys.H.diagonal().cwiseInverse();

    diagonal = vec::Zero(sys.n);
    
    // change diagonal compliance for unilateral constraints
    const real omega = this->omega.getValue();
    for(unsigned i = 0; i < sys.n; ++i) {

        if( unilateral(i) ) {

            diagonal(i) = sys.J.row(i).dot( Hdiag_inv.asDiagonal() * sys.J.row(i).transpose());
            // diagonal(i) = 1;

            sub.A.coeffRef(sub.P.cols() + i,
                           sub.P.cols() + i) = -omega * diagonal(i);
        }
    }

    // std::cout << unilateral.transpose() << std::endl;
    // std::cout << "Hinv " << Hdiag_inv.transpose() << std::endl;
    // std::cout << "diagonal " << diagonal.transpose() << std::endl;

    // factor the modified kkt
    sub.factor(*response);
}


// good luck with that :p
void ModulusSolver::solve(vec& res,
                          const AssembledSystem& sys,
                          const vec& rhs) const {

    vec constant = rhs;
    if( sys.n ) constant.tail(sys.n) = -constant.tail(sys.n);

    vec tmp;
    sub.solve(*response, tmp, constant);

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
        
        // prod is wrong, we need to add back 2 omega |z|
        sub.prod(tmp, tmp);
        
        // tmp += 2 * omega.getValue() * zabs;
        tmp.tail(sys.n).array() += (2 * omega) * diagonal.array() * zabs.array();

        // 
        tmp -= constant;

        // solve
        sub.solve(*response, tmp, tmp);

        // backup old dual
        old = y.tail(sys.n);
        
        y.head(sys.m).array() -= tmp.head(sys.m).array();
        y.tail(sys.n).array() -= tmp.tail(sys.n).array() + unilateral.array() * y.tail(sys.n).array();
        
        error = unilateral.cwiseProduct(old - y.tail(sys.n)).norm();
        if( error <= precision ) break;
    }

    sout << "fixed point error: " << error << "\titerations: " << k << sendl;

    // and that should be it ?
    res = y;

    // add |z| to dual to get lambda
    res.tail(sys.n).array() += unilateral.array() * y.tail(sys.n).array().abs();
    
}


}
}
}

