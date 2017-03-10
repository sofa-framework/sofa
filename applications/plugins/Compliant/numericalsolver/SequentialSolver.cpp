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
	: omega(initData(&omega, (SReal)1.0, "omega", "SOR parameter:  omega < 1 : better, slower convergence, omega = 1 : vanilla gauss-seidel, 2 > omega > 1 : faster convergence, ok for SPD systems, omega > 2 : will probably explode" ))
{}


BaseSequentialSolver::block::block() : offset(0), size(0), projector(0), activated(false) { }

void BaseSequentialSolver::fetch_blocks(const system_type& system) {

//    serr<<SOFA_CLASS_METHOD<<sendl;

    // TODO don't free memory ?
    blocks.clear();
	
	unsigned off = 0;

    for(unsigned i = 0, n = system.compliant.size(); i < n; ++i)
    {
		system_type::dofs_type* const dofs = system.compliant[i];
        const system_type::constraint_type& constraint = system.constraints[i];

        const unsigned dim = dofs->getDerivDimension();

        for(unsigned k = 0, max = dofs->getSize(); k < max; ++k)
        {
            block b;

            b.offset = off;
            b.size = dim;
            b.projector = constraint.projector.get();

            assert( !b.projector || !b.projector->mask || b.projector->mask->empty() || b.projector->mask->size() == max );
            b.activated = !b.projector || !b.projector->mask || b.projector->mask->empty() || (*b.projector->mask)[k];

            blocks.push_back( b );
            off += dim;
        }
    }

}


// TODO optimize remaining allocs
void BaseSequentialSolver::factor_block(inverse_type& inv, const schur_type& schur) {
	inv.compute( schur );

#ifndef NDEBUG
    if( inv.info() != Eigen::Success ){
        std::cerr << SOFA_CLASS_METHOD<<"non invertible block Schur." << std::endl;
        std::cerr << schur << std::endl;
    }
#endif
}

void BaseSequentialSolver::factor(const system_type& system) {

    fetch_blocks( system );
    factor_impl( system );
}

void BaseSequentialSolver::factor_impl(const system_type& system) {
	scoped::timer timer("system factorization");
 
	Benchmark::scoped_timer bench_timer(this->bench, &Benchmark::factor);

	// response matrix
	assert( response );


    SubKKT::projected_primal(sub, system);

    sub.factor(*response);
	
	// compute block responses
	const unsigned n = blocks.size();
	
    if( !n ) return;

	blocks_inv.resize( n );


    sub.solve_opt( *response, mapping_response, system.J );  // mapping_response = PHinv(JP)^T

    JP = system.J * system.P;

	// to avoid allocating matrices for each block, could be a vec instead ?
    dmat storage;

	// build blocks and factorize
	for(unsigned i = 0; i < n; ++i) {
		const block& b = blocks[i];
		
		// resize storage if needed TODO alloc max size only once
        if( (dmat::Index)b.size > storage.rows() ) storage.resize(b.size, b.size);
		
		// view on storage
		schur_type schur(storage.data(), b.size, b.size);
		
		// temporary sparse mat, difficult to remove :-/
        static cmat tmp; // try to improve matrix allocation
        tmp = JP.middleRows(b.offset, b.size) *
            mapping_response.middleCols(b.offset, b.size);
		
		// fill constraint block
        schur = tmp;
		
		// real symmetry = (schur - schur.transpose()).squaredNorm() / schur.size();
		// assert( std::sqrt(symmetry) < 1e-8 );
		
		// add diagonal C block
		for( unsigned r = 0; r < b.size; ++r) {
            for(rmat::InnerIterator it(system.C, b.offset + r); it; ++it) {
				
				// paranoia, i has it
				assert( it.col() >= int(b.offset) );
				assert( it.col() < int(b.offset + b.size) );
				
				schur(r, it.col() - int(b.offset)) += it.value();
			}
		}

        factor_block( blocks_inv[i], schur );
    }

}

// TODO make sure this does not cause any alloc
void BaseSequentialSolver::solve_block(chunk_type result, const inverse_type& inv, chunk_type rhs) const {
	assert( !has_nan(rhs.eval()) );
	result = rhs;

    bool ret = inv.solveInPlace(result);
    assert( !has_nan(result.eval()) );
    assert( ret );

	(void) ret;
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
                                 real /*damping*/ ) const {

	// TODO size asserts
	
	// error norm2 estimate (seems conservative and much cheaper to
	// compute anyways)
	real estimate = 0;

    SReal omega = this->omega.getValue();
		
	// inner loop
	for(unsigned i = 0, n = blocks.size(); i < n; ++i) {
			
		const block& b = blocks[i];
			 
        // data chunks
        chunk_type lambda_chunk(&lambda(b.offset), b.size);
        chunk_type delta_chunk(&delta(b.offset), b.size);

        // if the constraint is activated, solve it
        if( b.activated )
        {
            chunk_type error_chunk(&error(b.offset), b.size);

            // update rhs TODO track and remove possible allocs
            error_chunk.noalias() = rhs.segment(b.offset, b.size);
            error_chunk.noalias() -= JP.middleRows(b.offset, b.size) * net;
            error_chunk.noalias() -= sys.C.middleRows(b.offset, b.size) * lambda;

            // error estimate update, we sum current chunk errors
            // estimate += error_chunk.squaredNorm();

            // solve for lambda changes
            solve_block(delta_chunk, blocks_inv[i], error_chunk);

            // backup old lambdas
            error_chunk = lambda_chunk;

            // update lambdas
            lambda_chunk = lambda_chunk + omega * delta_chunk;

            // project new lambdas if needed
            if( b.projector ) {
                b.projector->project( lambda_chunk.data(), lambda_chunk.size(), i, correct );
                assert( !has_nan(lambda_chunk.eval()) );
            }

            // correct lambda differences based on projection
            delta_chunk = lambda_chunk - error_chunk;
        }
        else // deactivated constraint
        {
            // force lambda to be 0
            delta_chunk = -lambda_chunk;
            lambda_chunk.setZero();
        }

		// we estimate the total lambda change. since GS convergence
		// is linear, this can give an idea about current precision.
		estimate += delta_chunk.squaredNorm();

		// incrementally update net forces, we only do fresh
		// computation after the loop to keep perfs decent
        net.noalias() += mapping_response.middleCols(b.offset, b.size) * delta_chunk;
		// net.noalias() = mapping_response * lambda;

        // fix net to avoid error accumulations ?
	}

	// std::cerr << "sanity check: " << (net - mapping_response * lambda).norm() << std::endl;

	// TODO is this needed to avoid error accumulation ?
	// net = mapping_response * lambda;

	// TODO flag to return real residual estimate !! otherwise
	// convergence plots are not fair.
	return estimate;
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

	// reset bench if needed
	if( this->bench ) {
		bench->clear();
		bench->restart();
	}


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
    vec constant = rhs.tail(sys.n) - JP * free_res.head(sys.m);
	
	// lcp error
    vec error = vec::Zero(sys.n);
	
	const real epsilon = relative.getValue() ? 
		constant.norm() * precision.getValue() : precision.getValue();

	if( this->bench ) this->bench->lcp(sys, constant, *response, lambda);


	// outer loop
	unsigned k = 0, max = iterations.getValue();
//	vec primal;
	for(k = 0; k < max; ++k) {

        real estimate2 = step( lambda, net, sys, constant, error, delta, correct, damping );

		if( this->bench ) this->bench->lcp(sys, constant, *response, lambda);
		
		// stop if we only gain one significant digit after precision
		if( std::sqrt(estimate2) / sys.n <= epsilon ) break;
	}

    res.head( sys.m ) = free_res + net;
    res.tail( sys.n ) = lambda;



    if( this->f_printLog.getValue() )
        serr << "iterations: " << k << ", (abs) residual: " << (net - mapping_response * lambda).norm() << sendl;
	

}




//////////////




SOFA_DECL_CLASS(SequentialSolver)
int SequentialSolverClass = core::RegisterObject("Sequential Impulses solver").add< SequentialSolver >();






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
