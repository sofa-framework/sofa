#include "AssembledSystem.h"
#include <sofa/helper/rmath.h>

#include <iostream>

namespace sofa {
namespace component {
namespace linearsolver {


void AssembledSystem::reset(unsigned m, unsigned n) {
    this->m = m;
    this->n = n;
    this->dt = 0;

	if( !m ) return;

	H.resize(m, m); 
	P.resize(m, m); 
			
	if( n ) {
		J.resize(n, m);
		C.resize(n, n);
	}

    constraints.clear();
    master.clear();
    compliant.clear();
    isPIdentity = true;
}

AssembledSystem::AssembledSystem(unsigned m, unsigned n) 
	: m(m), 
	  n(n),
	  dt(0)
{
    reset(m, n);
}

unsigned AssembledSystem::size() const { return m + n; }
			

void AssembledSystem::debug(SReal /*thres*/) const {
	
	std::cerr << "H:" << std::endl
	          << H << std::endl
	          << "P:" << std::endl
	          << P << std::endl;
	if( n ) { 
			
		std::cerr << "J:" << std::endl 
		          << J << std::endl
		          << "C:" << std::endl
		          << C << std::endl;
	}

}



void AssembledSystem::copyFromMultiVec(vec &target, core::ConstMultiVecId derivId)
{
    assert(target.size() >= m);
    unsigned off = 0;
    // master dofs
    for(unsigned i = 0, end = master.size(); i < end; ++i)
    {
        dofs_type* dofs = master[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->copyToBuffer(&target(off), derivId.getId(dofs), dim);
        off += dim;
    }

}


void AssembledSystem::copyToMultiVec( core::MultiVecId targetId, const vec& source)
{
    unsigned off = 0;
    // master dofs
    for(unsigned i = 0, end = master.size(); i < end; ++i)
    {
        dofs_type* dofs = master[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->copyFromBuffer(targetId.getId(dofs), &source(off), dim);
        off += dim;
    }

}

void AssembledSystem::addToMultiVec( core::MultiVecId targetId, const vec& source )
{
    unsigned off = 0;
    // master dofs
    for(unsigned i = 0, end = master.size(); i < end; ++i)
    {
        dofs_type* dofs = master[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->addFromBuffer(targetId.getId(dofs), &source(off), dim);
        off += dim;
    }

}









void AssembledSystem::copyFromCompliantMultiVec(vec &target, core::ConstMultiVecId derivId)
{
    assert(target.size() >= n);
    unsigned off = target.size()==n ? 0 : m; // if target is of size m+n, only copy target.tail(n)
    // compliant dofs
    for(unsigned i = 0, end = compliant.size(); i < end; ++i)
    {
        dofs_type* dofs = compliant[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->copyToBuffer(&target(off), derivId.getId(dofs), dim);
        off += dim;
    }

}


void AssembledSystem::copyToCompliantMultiVec( core::MultiVecId targetId, const vec& source)
{
    unsigned off = source.size()==n ? 0 : m;
    // compliant dofs
    for(unsigned i = 0, end = compliant.size(); i < end; ++i)
    {
        dofs_type* dofs = compliant[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->copyFromBuffer(targetId.getId(dofs), &source(off), dim);
        off += dim;
    }

}

void AssembledSystem::addToCompliantMultiVec( core::MultiVecId targetId, const vec& source )
{
    unsigned off = source.size()==n ? 0 : m;
    // compliance dofs
    for(unsigned i = 0, end = compliant.size(); i < end; ++i)
    {
        dofs_type* dofs = compliant[i];

        unsigned dim = dofs->getMatrixSize();

        dofs->addFromBuffer(targetId.getId(dofs), &source(off), dim);
        off += dim;
    }

}



//static bool diagonal_dominant(const AssembledSystem& sys)
//{
//	typedef rmat::Index Index;

//	rmat PH = sys.P * sys.H;

//	typedef SReal real;

//    if( sys.n )
//    {
//        rmat PJt = sys.P * sys.J.transpose();

//        for( unsigned i = 0 ; i < sys.m ; ++i )
//	        {
//		        real d = helper::rabs(PH.coeff(i,i));

//		        real o = -d;
//		        for( Index j=0 ; j<PH.cols()  ; ++j ) o += helper::rabs(PH.coeff(i,j));
//		        for( Index j=0 ; j<PJt.cols() ; ++j ) o += helper::rabs(PJt.coeff(i,j));

//		        if( o > d ) return false;
//	        }

//        for( unsigned i=0 ; i< sys.n ; ++i )
//	        {
//		        real d = helper::rabs(sys.C.coeff(i,i));

//		        real o = -d;
//		        for( Index j=0 ; j<sys.C.cols() ; ++j ) o += helper::rabs(sys.C.coeff(i,j));
//		        for( Index j=0 ; j<sys.J.cols() ; ++j ) o += helper::rabs(sys.J.coeff(i,j));

//		        if( o > d ) return false;
//	        }
//    }
//    else
//	    {
//		    for( unsigned i=0 ; i< sys.m ; ++i )
//			    {
//				    real d = helper::rabs(PH.coeff(i,i));

//				    real o = -d;
//				    for( Index j=0 ; j<PH.cols() ; ++j ) o += helper::rabs(PH.coeff(i,j));

//				    if( o > d ) return false;
//			    }
//	    }

//    return true;
//}




}
}
}
