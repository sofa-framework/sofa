#include "SequentialSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "EigenSparseResponse.h"
#include "SubKKT.inl"

#include "../utils/scoped.h"
#include "../utils/nan.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(SequentialSolver)
int SequentialSolverClass = core::RegisterObject("Sequential Impulses solver").add< SequentialSolver >();


			

SequentialSolver::SequentialSolver() 
	: omega(initData(&omega, (SReal)1.0, "omega", "SOR parameter:  omega < 1 : better, slower convergence, omega = 1 : vanilla gauss-seidel, 2 > omega > 1 : faster convergence, ok for SPD systems, omega > 2 : will probably explode" ))
    , d_bilateralBlock(initData(&d_bilateralBlock, false, "bilateralBlock", "One big block for all bilateral constraints, slow for now" ))
{}


SequentialSolver::block::block() : offset(0), size(0), projector(0), activated(false) { }

void SequentialSolver::fetch_blocks(const system_type& system) {

    bool bilateralBlock = d_bilateralBlock.getValue();


	// TODO don't free memory ?
	blocks.clear();
    bilateral_block.size = 0;
    bilateral_block.chunks.clear();
	
	unsigned off = 0;

    for(unsigned i = 0, n = system.compliant.size(); i < n; ++i)
    {
		system_type::dofs_type* const dofs = system.compliant[i];
        const system_type::constraint_type& constraint = system.constraints[i];
        Constraint* proj = constraint.projector.get();

        if( proj || !bilateralBlock )
        {
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
        else
        {
            const unsigned dim = dofs->getMatrixSize();

            bilateral_block.chunks.push_back( Bilateral_block::chunk( off, dim ) );

            off += dim;
            bilateral_block.size += dim;
        }
    }

}


// TODO optimize remaining allocs
void SequentialSolver::factor_block(inverse_type& inv, const schur_type& schur) {
	inv.compute( schur );

#ifndef NDEBUG
    if( inv.info() == Eigen::NumericalIssue ){
        std::cerr << SOFA_CLASS_METHOD<<"block Schur is not psd. System solution will be wrong." << std::endl;
        std::cerr << schur << std::endl;
    }
#endif
}



void SequentialSolver::factor(const system_type& system) { 
	scoped::timer timer("system factorization");
 
	Benchmark::scoped_timer bench_timer(this->bench, &Benchmark::factor);

	// response matrix
	assert( response );

    SubKKT::projected_primal(sub, system);
    sub.factor(*response);

	// find blocks
	fetch_blocks(system);
	
	// compute block responses
	const unsigned n = blocks.size();
	
    if( !n && !bilateral_block.size ) return;

	blocks_inv.resize( n );

    sub.solve_filtered( *response, mapping_response, system.J, JP );  // mapping_response = PHinv(JP)^T

	
	// to avoid allocating matrices for each block, could be a vec instead ?
    dmat storage;

	// build blocks and factorize
	for(unsigned i = 0; i < n; ++i) {
		const block& b = blocks[i];
		
		// resize storage if needed TODO alloc max size only once
		if( b.size > storage.rows() ) storage.resize(b.size, b.size);
		
		// view on storage
		schur_type schur(storage.data(), b.size, b.size);
		
		// temporary sparse mat, difficult to remove :-/
		cmat tmp = JP.middleRows(b.offset, b.size) * 
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

    if( bilateral_block.size )
    {
        cmat schur( bilateral_block.size, bilateral_block.size );

        unsigned off = 0;

        for( unsigned i=0, iend=bilateral_block.chunks.size() ; i<iend ; ++i )
        {
            const unsigned offset = bilateral_block.chunks[i].first;
            const unsigned size = bilateral_block.chunks[i].second;

            cmat tmp = JP.middleRows(offset, size) * mapping_response.middleCols(offset, size)
                              + cmat( system.C.block( offset, offset, size , size ) );

            for( unsigned r = 0; r < size; ++r) {
                schur.startVec( off+r );
                for(cmat::InnerIterator it(tmp, r); it; ++it) {
                    schur.insertBack( off+it.row()-offset, off+r ) += it.value();
                }
            }

            off += size;
        }
        schur.finalize();

        bilateral_block.inv.compute( schur.triangularView< Eigen::Lower >() );
    }
}

// TODO make sure this does not cause any alloc
void SequentialSolver::solve_block(chunk_type result, const inverse_type& inv, chunk_type rhs) const {
	assert( !has_nan(rhs.eval()) );
	result = rhs;

    bool ret = inv.solveInPlace(result);
    assert( !has_nan(result.eval()) );
    assert( ret );

	(void) ret;
}



void SequentialSolver::init() {
	
    IterativeSolver::init();

	// let's find a response 
	response = this->getContext()->get<Response>( core::objectmodel::BaseContext::Local );

	// fallback in case we missed
	if( !response ) {
        response = new LDLTResponse();
        this->getContext()->addObject( response );
        std::cout << "SequentialSolver: fallback response class: "
                  << response->getClassName()
                  << " added to the scene" << std::endl;
	}

}


// this is where the magic happens
SReal SequentialSolver::step(vec& lambda,
                             vec& net, 
                             const system_type& sys,
                             const vec& rhs,
                             vec& error, vec& delta,
							 bool correct ) const {

	// TODO size asserts
	
	// error norm2 estimate (seems conservative and much cheaper to
	// compute anyways)
	real estimate = 0;


    // solve bilaterals first
    if( bilateral_block.size )
    {
        vec error( bilateral_block.size ), delta( bilateral_block.size );

        unsigned off = 0;
        for( unsigned i=0, iend=bilateral_block.chunks.size() ; i<iend ; ++i )
        {
            const unsigned offset = bilateral_block.chunks[i].first;
            const unsigned size = bilateral_block.chunks[i].second;

            error.segment(off, size) = rhs.segment(offset, size) - JP.middleRows(offset, size) * net
                                              - sys.C.middleRows(offset, size) * lambda;

           off += size;
        }

        delta = omega.getValue() * bilateral_block.inv.solve(error);

        estimate += delta.squaredNorm();

        off = 0;
        for( unsigned i=0, iend=bilateral_block.chunks.size() ; i<iend ; ++i )
        {
            const unsigned offset = bilateral_block.chunks[i].first;
            const unsigned size = bilateral_block.chunks[i].second;

            net.noalias() = net + mapping_response.middleCols(offset, size) * delta.segment(off, size);

            lambda.segment(offset, size) += delta.segment(off, size);

            off += size;
        }

    }

		
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
            error_chunk.noalias() = error_chunk	- JP.middleRows(b.offset, b.size) * net;
            error_chunk.noalias() = error_chunk - sys.C.middleRows(b.offset, b.size) * lambda;

            // error estimate update, we sum current chunk errors
            // estimate += error_chunk.squaredNorm();

            // solve for lambda changes
            solve_block(delta_chunk, blocks_inv[i], error_chunk);

            // backup old lambdas
            error_chunk = lambda_chunk;

            // update lambdas
            lambda_chunk = lambda_chunk + omega.getValue() * delta_chunk;

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
		net.noalias() = net + mapping_response.middleCols(b.offset, b.size) * delta_chunk;
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




void SequentialSolver::solve(vec& res,
							 const system_type& sys,
							 const vec& rhs) const {
	solve_impl(res, sys, rhs, false );
}


void SequentialSolver::correct(vec& res,
							   const system_type& sys,
							   const vec& rhs,
							   real /*damping*/ ) const {
	solve_impl(res, sys, rhs, true );
}


void SequentialSolver::solve_impl(vec& res,
								  const system_type& sys,
								  const vec& rhs,
								  bool correct) const {
	assert( response );

	// reset bench if needed
	if( this->bench ) {
		bench->clear();
		bench->restart();
	}


	// free velocity
    vec free_res( sub.size_sub() );
    sub.solve_filtered(*response, free_res, rhs.head(sys.m), SubKKT::PRIMAL);


    // we're done
    if( !sys.n )
    {
        res.head( sys.m ) = sub.unproject_primal(free_res);
        return;
    }

	
	// lagrange multipliers TODO reuse res.tail( sys.n ) ?
	vec lambda = res.tail(sys.n); 

	// net constraint velocity correction
	vec net = mapping_response * lambda;
	
	// lambda change work vector
	vec delta = vec::Zero( sys.n );
	
	// lcp rhs 
    vec constant = rhs.tail(sys.n) - JP * free_res;
	
	// lcp error
	vec error = vec::Zero( sys.n );
	
	const real epsilon = relative.getValue() ? 
		constant.norm() * precision.getValue() : precision.getValue();

	if( this->bench ) this->bench->lcp(sys, constant, *response, lambda);


	// outer loop
	unsigned k = 0, max = iterations.getValue();
//	vec primal;
	for(k = 0; k < max; ++k) {

        real estimate2 = step( lambda, net, sys, constant, error, delta, correct );

		if( this->bench ) this->bench->lcp(sys, constant, *response, lambda);
		
		// stop if we only gain one significant digit after precision
		if( std::sqrt(estimate2) / sys.n <= epsilon ) break;
	}

    res.head( sys.m ) = sub.unproject_primal( free_res + net );
    res.tail( sys.n ) = lambda;



    if( this->f_printLog.getValue() )
        serr << "iterations: " << k << ", (abs) residual: " << (net - mapping_response * lambda).norm() << sendl;
	

}

}
}
}
