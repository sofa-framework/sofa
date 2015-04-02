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
{}


SequentialSolver::block::block() : offset(0), size(0), projector(0), activated(false) { }

void SequentialSolver::fetch_blocks(const system_type& system) {

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

        real estimate2 = step( lambda, net, sys, constant, error, delta, correct );

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




SOFA_DECL_CLASS(PGSSolver)
int PGSSolverClass = core::RegisterObject("Sequential Impulses solver").add< PGSSolver >();






// build a projection basis based on constraint types
// TODO put it in a helper file that can be used by other solvers?
static unsigned projection_bilateral(AssembledSystem::rmat& Q_bilat, AssembledSystem::rmat& Q_unil, const AssembledSystem& sys)
{
    // flag which constraint are bilateral
    vector<bool> bilateral( sys.n );
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
            const vector<bool>::iterator itoff = bilateral.begin() + off;
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



bool PGSSolver::LocalSubKKT::projected_kkt_bilateral(rmat&H, rmat&P, rmat& Q, rmat&Q_star, const AssembledSystem& sys, real eps, bool only_lower)
{
    scoped::timer step("subsystem kkt-bilateral projection");

    projection_basis(P, sys.P, sys.isPIdentity);

    if(sys.n)
    {
        unsigned nb_bilaterals = projection_bilateral( Q, Q_star, sys );

        if( !nb_bilaterals ) // no bilateral constraints
        {
            H = rmat();
            return false;
        }
        else
        {
            filter_kkt(H, sys.H, P, Q, sys.J, sys.C, eps, sys.isPIdentity, nb_bilaterals == sys.n, only_lower);
            return true;
        }
    }
    else // no constraints
    {
        H = rmat();
        return false;
    }
}








void PGSSolver::factor(const system_type& system) {
    scoped::timer timer("system factorization");


    if( !LocalSubKKT::projected_kkt_bilateral( m_localSystem.H, m_localSub.P, m_localSub.Q, m_localSub.Q_unil, system, 1e-15 ) )
        return SequentialSolver::factor( system );


    m_localSystem.dt = system.dt;
    m_localSystem.m = m_localSystem.H.rows();
    m_localSystem.n = m_localSub.Q_unil.cols();
    m_localSystem.P.resize( m_localSystem.m, m_localSystem.m );
    m_localSystem.P.setIdentity();
    m_localSystem.isPIdentity = true;

    if( m_localSystem.n ) // there are non-bilat constraints
    {
        // keep only unilateral constraints in C
        m_localSystem.C.resize( m_localSystem.n, m_localSystem.n );
        m_localSystem.C = m_localSub.Q_unil.transpose() * system.C * m_localSub.Q_unil;


        // compute J_unil and resize it
        m_localSystem.J.resize( m_localSystem.n, m_localSystem.m );
        rmat tmp = m_localSub.Q_unil.transpose() * system.J * m_localSub.P;
        for(unsigned i = 0; i < tmp.rows(); ++i)
        {
            m_localSystem.J.startVec( i );
            for(rmat::InnerIterator it(tmp, i); it; ++it) {
                m_localSystem.J.insertBack(i, it.col()) = it.value();
            }
        }
        m_localSystem.J.finalize();
    }
    else
    {
        m_localSystem.J = rmat();
        m_localSystem.C = rmat();
    }


   fetch_unilateral_blocks( system );


   SequentialSolver::factor( m_localSystem );
}


void PGSSolver::solve_impl(vec& res,
                           const system_type& sys,
                           const vec& rhs,
                           bool correct) const {

    if( !m_localSystem.H.nonZeros() )
        return SequentialSolver::solve_impl( res, sys, rhs, correct );
    assert( m_localSystem.m == m_localSub.P.cols() + m_localSub.Q.cols() );
    assert( m_localSystem.n == m_localSub.Q_unil.cols() );
    const size_t localsize = m_localSystem.size();

    vec localrhs(localsize), localres(localsize);

    // reordering rhs
    localrhs.head( m_localSub.P.cols() ) = m_localSub.P.transpose() * rhs.head( sys.m ); // primal
    localrhs.segment( m_localSub.P.cols(), m_localSub.Q.cols() ) = - m_localSub.Q.transpose() * rhs.tail( sys.n ); // bilaterals
    if( m_localSub.Q_unil.cols() )
        localrhs.tail( m_localSub.Q_unil.cols() ) = m_localSub.Q_unil.transpose() * rhs.tail( sys.n ); // other constraints

    SequentialSolver::solve_impl( localres, m_localSystem, localrhs, correct );


    // reordering res
    res.head( sys.m ) = m_localSub.P * localres.head( m_localSub.P.cols() ); // primal
    res.tail( sys.n ) = m_localSub.Q * localres.segment( m_localSub.P.cols(), m_localSub.Q.cols() ); // bilaterals
    if( m_localSub.Q_unil.cols() )
        res.tail( sys.n ) += m_localSub.Q_unil * localres.tail( m_localSub.Q_unil.cols() ); // other constraints
}



// the only difference with the regular implementation is to consider only non-bilateral constraints
// overloading this function allows no to set m_localSystem.master/compliance/constraints
void PGSSolver::fetch_unilateral_blocks(const system_type& system)
{
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

void PGSSolver::fetch_blocks(const system_type& system)
{
}


}
}
}
