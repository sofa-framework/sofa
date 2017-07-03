#include "SequentialSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "EigenSparseResponse.h"
#include "SubKKT.inl"

#include "../utils/scoped.h"
#include "../utils/nan.h"


namespace sofa {
namespace component {
namespace linearsolver {



BaseSequentialSolver::BaseSequentialSolver()
	: omega(initData(&omega, (SReal)1.0, "omega",
                     "SOR parameter:  omega < 1 : better, slower convergence, omega = 1 : vanilla gauss-seidel, 2 > omega > 1 : faster convergence, ok for SPD systems, omega > 2 : will probably explode" )),
      paranoia(initData(&paranoia, false, "paranoia", "add paranoid steps to counter numerical issues")) 
      
{}


BaseSequentialSolver::block::block() : offset(0), size(0), projector(0), activated(false) { }

void BaseSequentialSolver::fetch_blocks(const system_type& system) {
    
//    serr<<SOFA_CLASS_METHOD<<sendl;

    // TODO don't free memory ?
    blocks.clear();
	
	unsigned off = 0;

    for(unsigned i = 0, n = system.compliant.size(); i < n; ++i) {
		system_type::dofs_type* const dofs = system.compliant[i];
        const system_type::constraint_type& constraint = system.constraints[i];

        const unsigned dim = dofs->getDerivDimension();

        for(unsigned k = 0, max = dofs->getSize(); k < max; ++k) {
            block b;

            b.offset = off;
            b.size = dim;
            b.projector = constraint.projector.get();

            assert( !b.projector || !b.projector->mask || 
                    b.projector->mask->empty() || b.projector->mask->size() == max );
            
            b.activated = !b.projector || !b.projector->mask
              || b.projector->mask->empty() || (*b.projector->mask)[k];
            
            blocks.push_back( b );
            off += dim;
        }
    }

}




void BaseSequentialSolver::factor(const system_type& system) {

    fetch_blocks( system );
    factor_impl( system );
}


template<class LHSIterator, class RHSIterator>
static inline SReal sparse_dot(LHSIterator&& lhs, RHSIterator&& rhs) {

    SReal res = 0;

    while(lhs && rhs) {

        if( rhs.index() < lhs.index() ) {
            ++rhs;
        } else if (rhs.index() > lhs.index()) {
            ++lhs;
        } else {
            res += lhs.value() * rhs.value();
            
            ++lhs;
            ++rhs;
        }
    }
    
    return res;
}



template<class RowSparse, class ColSparse>
static inline SReal sparse_dot(const RowSparse& lhs, unsigned i,
                               const ColSparse& rhs, unsigned j) {
    typename RowSparse::InnerIterator it_lhs(lhs, i);
    typename ColSparse::InnerIterator it_rhs(rhs, j);

    return sparse_dot(std::move(it_lhs), std::move(it_rhs));
}



template<class ColDenseMatrix, class RowSparse, class ColSparse>
static void add_sparse_product_to_dense(ColDenseMatrix& out, 
                                        const RowSparse& lhs, const ColSparse& rhs,
                                        bool only_upper) {
    assert(lhs.cols() == rhs.rows());

    // TODO static_assert RowSparse is row-major && ColSparse is col-major
    for(unsigned j = 0, cols = rhs.cols(); j < cols; ++j) {
        for(typename ColSparse::InnerIterator it_rhs(rhs, j); it_rhs; ++it_rhs) {
            const unsigned k = it_rhs.index();
            const SReal v = it_rhs.value();
            
            for(typename RowSparse::InnerIterator it_lhs(lhs, k);
                it_lhs; ++it_lhs) {
                const unsigned i = it_lhs.index();
                if(only_upper && (i > j)) break;
                out(i, j) += v * it_lhs.value();
            }
        }
    }

}

void BaseSequentialSolver::factor_impl(const system_type& system) {
    // fill and factor sub-kkt system
    SubKKT::projected_primal(sub, system);

	assert( response );    
    sub.factor(*response);
	
	// compute block responses
	const unsigned n = blocks.size();
    
    if( !n ) return;
    
    diagonal.resize( system.n );

    // project constraint matrix
    JP = system.J * system.P;

    // compute mapping_response = Hinv * J^T
    sub.solve_opt( *response, mapping_response, JP );  
    

	// build blocks and factorize
	for(unsigned i = 0; i < n; ++i) {
		const block& b = blocks[i];
		
		for( unsigned r = 0; r < b.size; ++r) {

            const unsigned off = b.offset + r;
            
            SReal& d = diagonal(off);
            d = sparse_dot(JP, off, mapping_response, off);

            // TODO is this the fastest?
            const SReal c = system.C.coeff(off, off);
            
            d += c;
            
            // sanity check
            if( d <= 0 ){
                
                // empty constraint row, this is an error
                const SReal norm = system.J.row(off).norm();
                if( norm == 0 ) {
                    serr << "zero constraint row: " << off << sendl;
                    simulation::Node* node = 0;
                    
                    if(b.projector) {
                        node = dynamic_cast<simulation::Node*>(b.projector->getContext());
                    } else {
                        unsigned tmp = 0;
                        for(const auto& c : system.compliant) {
                            const unsigned size = c->getMatrixSize();
                            if( (tmp + size) > off) {
                                node = dynamic_cast<simulation::Node*>(c->getContext());
                                break;
                            } else {
                                tmp += size;
                            }
                        }
                        assert( tmp == system.n );
                        assert( off < system.n );
                    }
                    
                    assert(node);
                    if( node ) {
                        auto mapping = node->mechanicalMapping.get();
                        serr << "mapping: " << mapping->getPathName() << sendl;
                    }
                    
                    assert(false && "empty constraint row");
                }

                // constraint row got zero-ed after fixed-constraint projection:
                // flag as disabled
                assert( c >= 0 );
                assert(JP.row(off).norm() == 0);
                
                d = -1;
            }
            
		}

        // TODO homogenize friction constraints
        
    }

    if(debug.getValue() ) {
        serr << "diagonal: " << diagonal.transpose() << sendl;
    }
}





void BaseSequentialSolver::init() {
	
    IterativeSolver::init();

	// let's find a response 
	response = this->getContext()->get<Response>( core::objectmodel::BaseContext::Local );

	// fallback in case we missed
	if( !response ) {
        response = new LDLTResponse();
        response->setName("response");
        this->getContext()->addObject( response );
        serr << "fallback Response: "
                  << response->getClassName()
                  << " added to the scene" << sendl;
	}

}


// this is where the magic happens
SReal BaseSequentialSolver::step(vec& lambda,
                                 vec& net, 
                                 const system_type& sys,
                                 const vec& rhs,
                                 vec& error, vec& delta,
                                 bool correct,
                                 real damping ) const {

    // fix-point error squared-norm: ||f(x) - x||^2
    real error_squared = 0;

    SReal omega = this->omega.getValue();
		
	// inner loop
	for(unsigned i = 0, n = blocks.size(); i < n; ++i) {
			
		const block& b = blocks[i];
			 
        // data chunks
        chunk_type lambda_chunk(&lambda(b.offset), b.size);
        chunk_type delta_chunk(&delta(b.offset), b.size);

        // if the constraint is activated, solve it
        if( b.activated ) {
            chunk_type error_chunk(&error(b.offset), b.size);

            // update error
            error_chunk.noalias() = rhs.segment(b.offset, b.size);
            error_chunk.noalias() -= JP.middleRows(b.offset, b.size) * net;
            error_chunk.noalias() -= sys.C.middleRows(b.offset, b.size) * lambda;
            error_chunk.noalias() -= damping * lambda_chunk;
            
            const const_chunk_type diag_chunk(&diagonal(b.offset), b.size);
            
            // solve for lambda changes w/ jacobi iteration
            delta_chunk = error_chunk.array() / (diag_chunk.array() + damping);
            assert( !has_nan(delta_chunk) );

            // handle disabled constraints
            delta_chunk.array() *= (diag_chunk.array() > 0).template cast<SReal>();
            
            // backup old lambdas
            error_chunk = lambda_chunk;

            // update lambdas
            lambda_chunk += omega * delta_chunk;

            // project new lambdas if needed
            if( b.projector ) {
                
                b.projector->project( lambda_chunk.data(), lambda_chunk.size(), i, correct );
                assert( !has_nan(lambda_chunk) );

            }

            // recompute deltas from projected lambdas
            delta_chunk = lambda_chunk - error_chunk;

        } else {
            // deactivated constraint: force lambda to be 0
            delta_chunk = -lambda_chunk;
            lambda_chunk.setZero();
        }

        // update fixpoint error
		error_squared += delta_chunk.squaredNorm();

		// incrementally update net forces, we only do fresh
		// computation after the loop to keep perfs decent
        net.noalias() += mapping_response.middleCols(b.offset, b.size) * delta_chunk;
		// net.noalias() = mapping_response * lambda;

        // fix net to avoid error accumulations ?
	}

	// std::cerr << "sanity check: " << (net - mapping_response * lambda).norm() << std::endl;

	// TODO is this needed to avoid error accumulation? apparently not, but
	// would be nice to have a paranoid flag to control it

    if(paranoia.getValue()) {
        net.noalias() = mapping_response * lambda;
    }
    
	return error_squared;
}

void BaseSequentialSolver::solve(vec& res,
                                 const system_type& sys,
                                 const vec& rhs) const {
    solve_impl(res, sys, rhs, false );
}


void BaseSequentialSolver::correct(vec& res,
                                   const system_type& sys,
                                   const vec& rhs,
                                   real damping ) const {
    solve_impl(res, sys, rhs, true, damping );
}

void BaseSequentialSolver::solve_impl(vec& res,
                                      const system_type& sys,
                                      const vec& rhs,
                                      bool correct,
                                      real damping) const {
	assert( response );

	// free velocity
    vec free_res( sys.m );
    sub.solve(*response, free_res, rhs.head( sys.m ));


    // we're done
    if( blocks.empty() )
    {
        res = free_res;
        return;
    }


    // lagrange multipliers TODO reuse res.tail( sys.n ) ?
    vec lambda = res.tail(sys.n);

	// net constraint velocity correction
	vec net = mapping_response * lambda;
	
	// lambda change work vector
    vec delta = vec::Zero(sys.n);
	
	// lcp rhs 
    const vec constant = rhs.tail(sys.n) - JP * free_res.head(sys.m);
	
	// lcp error
    vec error = vec::Zero(sys.n);
	
	const real epsilon = relative.getValue() ? 
		constant.norm() * precision.getValue() : precision.getValue();

	// outer loop
	unsigned k = 0, max = iterations.getValue();
	vec old;

    // convergence error
    SReal cv = 0;
    
	for(k = 0; k < max; ++k) {
        old = lambda;
        
        const SReal cv_squared = step( lambda, net, sys, constant, error, delta, correct, damping );

        cv = std::sqrt( cv_squared );
		if( cv < epsilon ) break;
	}
    
    res.head( sys.m ) = free_res + net;
    res.tail( sys.n ) = lambda;

    if(debug.getValue() ) {
        serr << "lcp pass: " << (correct ? "correct" : "dynamics") << sendl;
        serr << "lcp rhs: " << constant.transpose() << sendl;
        serr << "lcp lambda: " << lambda.transpose() << sendl;

        // // only make sense when all constraints are unilateral
        // const SReal unilateral_error =
        //     (JP * (mapping_response * lambda)
        //      + sys.C * lambda
        //      + damping * lambda
        //      - constant).array().min(lambda.array()).matrix().norm();
        
        // serr << "unilateral error: " << unilateral_error << sendl;
        
        serr << sendl;
    }

    
    if( this->f_printLog.getValue() ) {
        serr << "iterations: " << k << ", error: " << cv << sendl;
    }

}




//////////////




SOFA_DECL_CLASS(SequentialSolver)
static int SequentialSolverClass = core::RegisterObject("Sequential Impulses solver").add< SequentialSolver >();




// build a projection basis based on constraint types (bilateral vs others)
// TODO put it in a helper file that can be used by other solvers?
static unsigned projection_bilateral(AssembledSystem::rmat& Q_bilat, AssembledSystem::rmat& Q_unil, const AssembledSystem& sys)
{
    // flag which constraint are bilateral
    static helper::vector<bool> bilateral;
    bilateral.resize( sys.n );
    unsigned nb_bilaterals = 0;

    for(unsigned i=0, off=0, n=sys.compliant.size(); i < n; ++i)
    {
        const AssembledSystem::dofs_type* dofs = sys.compliant[i];
        const AssembledSystem::constraint_type& constraint = sys.constraints[i];

        const unsigned dim = dofs->getDerivDimension();

        for(unsigned k = 0, max = dofs->getSize(); k < max; ++k)
        {
            bool bilat = !constraint.projector.get(); // flag bilateral or not
            if( bilat ) nb_bilaterals += dim;
            const helper::vector<bool>::iterator itoff = bilateral.begin() + off;
            std::fill( itoff, itoff+dim, bilat );
            off += dim;
        }
    }

    if( !nb_bilaterals )  // no bilateral constraints
    {
        return 0;
//        Q_bilat = AssembledSystem::rmat();
//        Q_unil.resize(sys.n, sys.n);
//        Q_unil.setIdentity();
    }
    else if( nb_bilaterals == sys.n ) // every constraints are bilateral,
    {
        Q_bilat.resize(sys.n, sys.n);
        Q_bilat.setIdentity();
        Q_unil = AssembledSystem::rmat();
    }
    else
    {
        Q_bilat.resize( sys.n, nb_bilaterals );
        Q_bilat.reserve(nb_bilaterals);

        unsigned nb_unilaterals = sys.n-nb_bilaterals;
        Q_unil.resize( sys.n, nb_unilaterals );
        Q_unil.reserve(nb_unilaterals);

        unsigned off_bilat = 0;
        unsigned off_unil = 0;
        for( unsigned i = 0 ; i < sys.n ; ++i )
        {
            Q_bilat.startVec(i);
            Q_unil.startVec(i);

            if( bilateral[i] ) Q_bilat.insertBack(i, off_bilat++) = 1;
            else Q_unil.insertBack(i, off_unil++) = 1;
        }

        Q_bilat.finalize();
        Q_unil.finalize();
    }

    return nb_bilaterals;
}





bool SequentialSolver::LocalSubKKT::projected_primal_and_bilateral( AssembledSystem& res,
                                                              const AssembledSystem& sys,
                                                              real eps,
                                                              bool only_lower)
{
    scoped::timer step("subsystem primal-bilateral");

    projection_basis(P, sys.P, sys.isPIdentity);

    if(sys.n)
    {
        unsigned nb_bilaterals = projection_bilateral( Q, Q_unil, sys );

        if( !nb_bilaterals ) // no bilateral constraints
        {
            res.H = rmat();
            return false;
        }
        else
        {
            filter_kkt(res.H, sys.H, P, Q, sys.J, sys.C, eps, sys.isPIdentity, nb_bilaterals == sys.n, only_lower);

            res.dt = sys.dt;
            res.m = res.H.rows();
            res.n = Q_unil.cols();
            res.P.resize( res.m, res.m );
            res.P.setIdentity();
            res.isPIdentity = true;

            if( res.n ) // there are non-bilat constraints
            {
                // keep only unilateral constraints in C
                res.C.resize( res.n, res.n );
                res.C = Q_unil.transpose() * sys.C * Q_unil;

                // compute J_unil and resize it
                res.J.resize( res.n, res.m );

                static rmat tmp; // try to improve matrix allocation

                tmp = Q_unil.transpose() * sys.J * P;
                for( rmat::Index i = 0; i < tmp.rows(); ++i)
                {
                    res.J.startVec( i );
                    for(rmat::InnerIterator it(tmp, i); it; ++it) {
                        res.J.insertBack(i, it.col()) = it.value();
                    }
                }
                res.J.finalize();
            }
            else
            {
                res.J = rmat();
                res.C = rmat();
            }

            return true;
        }
    }
    else // no constraints
    {
        res.H = rmat();
        return false;
    }
}



void SequentialSolver::LocalSubKKT::toLocal( vec& local, const vec& global ) const
{
    assert( local.size() == P.cols() + Q.cols() + Q_unil.cols() );
    assert( global.size() == P.rows() + Q.rows() );

    local.head( P.cols() ) = P.transpose() * global.head( P.rows() ); // primal
    local.segment( P.cols(), Q.cols() ) = -Q.transpose() * global.tail( Q.rows() ); // bilaterals
    if( Q_unil.cols() )
        local.tail( Q_unil.cols() ) = Q_unil.transpose() * global.tail( Q.rows() ); // other constraints
}

void SequentialSolver::LocalSubKKT::fromLocal( vec& global, const vec& local ) const
{
    assert( local.size() == P.cols() + Q.cols() + Q_unil.cols() );
    assert( global.size() == P.rows() + Q.rows() );

    // TODO optimize copy of localres into res (constraints could be written in one loop)

    global.head( P.rows() ) = P * local.head( P.cols() ); // primal
    global.tail( Q.rows() ) = Q * local.segment( P.cols(), Q.cols() ); // bilaterals
    if( Q_unil.cols() )
        global.tail( Q.rows() ) += Q_unil * local.tail( Q_unil.cols() ); // other constraints
}


SequentialSolver::SequentialSolver()
    : d_iterateOnBilaterals(initData(&d_iterateOnBilaterals, true, "iterateOnBilaterals", "Should the bilateral constraint must be solved iteratively or factorized with the dynamics?"))
    , d_regularization(initData(&d_regularization, std::numeric_limits<SReal>::epsilon(), "regularization", "Optional diagonal Tikhonov regularization on bilateral constraints"))
{}


void SequentialSolver::factor(const system_type& system) {
    scoped::timer timer("system factorization");

    if( d_iterateOnBilaterals.getValue() ||
            !m_localSub.projected_primal_and_bilateral( m_localSystem, system, d_regularization.getValue(), response->isSymmetric() ) // no bilaterals
            )
    {
        fetch_blocks( system ); // find blocks
        factor_impl( system );
    }
    else
    {
        fetch_unilateral_blocks( system ); // find unilateral blocks
        factor_impl( m_localSystem );
    }
}




void SequentialSolver::solve(vec& res,
                             const system_type& sys,
                             const vec& rhs) const {
    solve_local(res, sys, rhs, false );
}


void SequentialSolver::correct(vec& res,
                               const system_type& sys,
                               const vec& rhs,
                               real damping ) const {
    solve_local(res, sys, rhs, true, damping );
}



void SequentialSolver::solve_local(vec& res,
                                   const system_type& sys,
                                   const vec& rhs,
                                   bool correct, 
                                   real damping) const {

    if( d_iterateOnBilaterals.getValue() || !m_localSystem.H.nonZeros() )
        return solve_impl( res, sys, rhs, correct, damping );

    if( d_iterateOnBilaterals.getValue() || !m_localSystem.H.nonZeros() ) {
        return solve_impl( res, sys, rhs, correct, damping );
    }
    
    const size_t localsize = m_localSystem.size();

    vec localrhs(localsize), localres(localsize);

    // reordering rhs
    m_localSub.toLocal( localrhs, rhs );
    // reordering res, for warm_start...
    m_localSub.toLocal( localres, res );

    // performing the solve on the reorganized system
    solve_impl( localres, m_localSystem, localrhs, correct, damping );

    // reordering res
    m_localSub.fromLocal( res, localres );
}



// the only difference with the regular implementation is to consider only non-bilateral constraints
void SequentialSolver::fetch_unilateral_blocks(const system_type& system)
{
//    serr<<SOFA_CLASS_METHOD<<sendl;

    // TODO don't free memory ?
    blocks.clear();

    unsigned off = 0;

    for(unsigned i = 0, n = system.compliant.size(); i < n; ++i)
    {
        system_type::dofs_type* const dofs = system.compliant[i];
        const system_type::constraint_type& constraint = system.constraints[i];

        const unsigned dim = dofs->getDerivDimension();

        Constraint* proj = constraint.projector.get();

        if( proj )
        {
            for(unsigned k = 0, max = dofs->getSize(); k < max; ++k)
            {
                block b;

                b.offset = off;
                b.size = dim;
                b.projector = proj;

                assert( !b.projector || !b.projector->mask || b.projector->mask->empty() || b.projector->mask->size() == max );
                b.activated = !b.projector || !b.projector->mask || b.projector->mask->empty() || (*b.projector->mask)[k];
                blocks.push_back( b );
                off += dim;
            }
        }
    }

    assert( off == m_localSystem.n );

}


}
}
}
