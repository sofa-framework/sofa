#ifndef COMPLIANT_SUBKKT_H
#define COMPLIANT_SUBKKT_H

#include "../initCompliant.h"
#include "../assembly/AssembledSystem.h"

namespace sofa {
namespace component {
namespace linearsolver {

class Response;

class SubKKT {
    typedef AssembledSystem::mat mat;
    typedef AssembledSystem::cmat cmat;
    typedef AssembledSystem::vec vec;

    // primal/dual selection matrices
    mat P, Q;

    // filtered subsystem
    mat A;

    mutable vec tmp1, tmp2;
public:

    static mat projection_basis(const mat& P);

    // standard projected (1, 1) schur subsystem
    SubKKT(const AssembledSystem& system);
    
    // TODO with dual projection matrix
    
    void factor(Response& response) const;

    void solve(Response& response, vec& result, const vec& rhs) const;
    void solve(Response& response, cmat& result, const cmat& rhs) const; 
    
};



}
}
}



#endif
