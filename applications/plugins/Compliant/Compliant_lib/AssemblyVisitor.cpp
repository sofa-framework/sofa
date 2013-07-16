#include "AssemblyVisitor.h"

#include <sofa/component/linearsolver/EigenVectorWrapper.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>

#include "./utils/scoped.h"
#include "./utils/find.h"
#include "./utils/cast.h"
#include "./utils/sparse.h"

namespace sofa {
namespace simulation {

using namespace component::linearsolver;
using namespace core::behavior;

typedef EigenVectorWrapper<SReal> wrap;

// some helpers
template<class Matrix>
bool zero(const Matrix& m) {
	return !m.nonZeros();
}
		
template<class Matrix>
bool empty(const Matrix& m) {
	return !m.rows();
}
		

template<class LValue, class RValue>
static void add(LValue& lval, const RValue& rval) {
	if( empty(lval) ) {
		lval = rval;
	} else {
		// paranoia, i has it
		lval += rval;
	}
}


static std::string pretty(AssemblyVisitor::dofs_type* dofs) {
	return dofs->getContext()->getName() + " / " + dofs->getName();
}

		
// right-shift matrix, size x (off + size) matrix: (0, id)
static AssemblyVisitor::mat shift_right(unsigned off, unsigned size, unsigned total_cols, SReal value = 1.0 ) {
	AssemblyVisitor::mat res( size, total_cols); 
	assert( total_cols >= (off + size) );
	
	res.reserve( size );
	
	for(unsigned i = 0; i < size; ++i) {
		res.startVec( i );
		res.insertBack(i, off + i) = value;
	}
	res.finalize();
	
	return res;
}


		
		
AssemblyVisitor::AssemblyVisitor(const core::MechanicalParams* mparams) 
	: base( mparams ),
	  mparams( mparams ),
	  start_node(0)

{ }
		
AssemblyVisitor::chunk::chunk() 
	: offset(0), 
	  size(0), 
	  damping(0), 
	  mechanical(false), 
	  vertex(-1), 
	  dofs(0) {
	
}

		
// convert a basematrix to a sparse matrix. TODO move this somewhere else ?
AssemblyVisitor::mat AssemblyVisitor::convert( const defaulttype::BaseMatrix* m) {
	assert( m );
			
	typedef EigenBaseSparseMatrix<double> matrixd;
	
	const matrixd* smd = dynamic_cast<const matrixd*> (m);
	if ( smd ) return smd->compressedMatrix.cast<SReal>();
			
	typedef EigenBaseSparseMatrix<float> matrixf;
			
	const matrixf* smf = dynamic_cast<const matrixf*>(m);
	if( smf ) return smf->compressedMatrix.cast<SReal>();

	
	std::cerr << "warning: slow matrix conversion" << std::endl;
			
	mat res(m->rowSize(), m->colSize());
	
	res.reserve(res.rows() * res.cols());
	for(unsigned i = 0, n = res.rows(); i < n; ++i) {
		res.startVec( i );
		for(unsigned j = 0, k = res.cols(); j < k; ++j) {
			res.insertBack(i, j) = m->element(i, j);
		}
	}

	return res;
}
		
// this is not thread safe
void AssemblyVisitor::vector(dofs_type* dofs, core::VecId id, const vec::ConstSegmentReturnType& data) {
    assert( dofs->getMatrixSize() == data.size() );
			
	// realloc
	if( data.size() > tmp.size() ) tmp.resize(data.size() );
	tmp.head(data.size()) = data;
	
	unsigned off = 0;
	wrap w(tmp);
	dofs->copyFromBaseVector(id, &w, off);
	assert( off == data.size() );
}		
		

AssemblyVisitor::vec AssemblyVisitor::vector(dofs_type* dofs, core::ConstVecId id) {
	unsigned size = dofs->getMatrixSize();

	const bool fast = false;
			
	if( fast ) {
		// map data
		const void* data = dofs->baseRead(id)->getValueVoidPtr();
		
		// TODO need to know if we're dealing with double or floats
		return Eigen::Map<const vec>( reinterpret_cast<const double*>(data), size);
	} else {
		
		// realloc
		if( size > tmp.size() ) tmp.resize(size);
		
		unsigned off = 0;
		wrap w(tmp);
		
		dofs->copyToBaseVector(&w , id, off);
				
		return tmp.head(size);
	}
}


// pretty prints a mapping
static inline std::string mapping_name(simulation::Node* node) {
	return node->mechanicalMapping->getName() + " (class: " + node->mechanicalMapping->getClassName() + ") ";
}
		
// mapping informations as a map (parent dofs -> (J, K) matrices )
AssemblyVisitor::chunk::map_type AssemblyVisitor::mapping(simulation::Node* node) {
	chunk::map_type res;
			
	if( !node->mechanicalMapping ) return res;
			
	using helper::vector;
			
	const vector<sofa::defaulttype::BaseMatrix*>* js = node->mechanicalMapping->getJs();
	const vector<sofa::defaulttype::BaseMatrix*>* ks = node->mechanicalMapping->getKs();
	
	assert( node->mechanicalMapping->getTo().size() == 1 && 
	        "only n -> 1 mappings are handled");
	
	vector<core::BaseState*> from = node->mechanicalMapping->getFrom();
	
	dofs_type* to = safe_cast<dofs_type>(node->mechanicalMapping->getTo()[0]);
//	unsigned rows = to->getMatrixSize();
	
	for( unsigned i = 0, n = from.size(); i < n; ++i ) {

		// parent dofs
		dofs_type* p = safe_cast<dofs_type>(from[i]);  
		
		// skip non-mechanical dofs
		if(!p) continue;
				
		// mapping wrt p
		chunk::mapped& c = res[p];

		if( js ) c.J = convert( (*js)[i] );
				
		if( empty(c.J) ) {
//			unsigned cols = p->getMatrixSize();
			
			std::string msg("empty mapping block for " + mapping_name(node) + " (is mapping matrix assembled ?)" );
			assert( false ); 
			
			throw std::logic_error(msg);
		}
				
		if( ks ) {
			c.K = convert( (*ks)[i] );
					
			// sanity check
			if( zero(c.K) ) {
				// TODO derp ?
				// std::cerr << mapping_name(node) << " has no geometric stiffness" << std::endl;
				// throw std::logic_error("empty geometric stiffness block for mapping " + mapping_name(node) );
			}
		} 
				
	}

	return res;
}


 
// mass matrix
AssemblyVisitor::mat AssemblyVisitor::mass(simulation::Node* node) {
	unsigned size = node->mechanicalState->getMatrixSize();

	mat res(size, size);
	
	if( !node->mass ) return res;
	
	typedef EigenBaseSparseMatrix<SReal> Sqmat;
	Sqmat sqmat( size, size );

	{
		SingleMatrixAccessor accessor( &sqmat );
		node->mass->addMToMatrix( mparams, &accessor );
	}
	
	sqmat.compress();
	res = sqmat.compressedMatrix.selfadjointView<Eigen::Upper>();
	return res;
}
		

// projection matrix
AssemblyVisitor::mat AssemblyVisitor::proj(simulation::Node* node) {
	assert( node->mechanicalState );
			
	unsigned size = node->mechanicalState->getMatrixSize();
			
	// identity matrix TODO alloc ?
	tmp_p.compressedMatrix = shift_right(0, size, size);
			
	for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++){
		node->projectiveConstraintSet[i]->projectMatrix(&tmp_p, 0);
	}

	return tmp_p.compressedMatrix;
}


// compliance matrix
AssemblyVisitor::mat AssemblyVisitor::compliance(simulation::Node* node) {
			
	mat res;
			
	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
					
		const BaseMatrix* c = ffield->getComplianceMatrix(mparams);
				
		if( c ) return convert( c );
				
	}
			
	return res;
}

		
// stiffness matrix
AssemblyVisitor::mat AssemblyVisitor::stiff(simulation::Node* node) {
			
	mat res;
	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
				
		const BaseMatrix* k = ffield->getStiffnessMatrix(mparams);
		
		if( k ) {
			add(res, convert( k ));
		} else {
			// std::cerr << ffield->getName() << " has no stiffness matrix lol" << std::endl;
		}
				
	}
	
	return res;
}


// fetches force
AssemblyVisitor::vec AssemblyVisitor::force(simulation::Node* node) {
	assert( node->mechanicalState );
	return vector(node->mechanicalState, core::VecDerivId::force());	
}


// fetches velocity
AssemblyVisitor::vec AssemblyVisitor::vel(simulation::Node* node) {
	assert( node->mechanicalState );
	return vector(node->mechanicalState, core::VecDerivId::velocity());	
}

// fetches damping term
AssemblyVisitor::real AssemblyVisitor::damping(simulation::Node* node) {
	assert( node->mechanicalState );
			
	assert( node->forceField.size() <= 1 );

	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
		return ffield->getDampingRatio();
	}
	
	return 0;
}

// fetches phi term
AssemblyVisitor::vec AssemblyVisitor::phi(simulation::Node* node) {
	assert( node->mechanicalState );
	assert( node->forceField.size() <= 1 );
		
	vec res;
	
	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
				
		if( ffield->getComplianceMatrix(mparams) ) { 
			ffield->writeConstraintValue( mparams, core::VecDerivId::force() );
			return force( node );
		}
	}
			
	return res;
}


AssemblyVisitor::vec AssemblyVisitor::lambda(simulation::Node* node) {
	assert( node->mechanicalState );

	// TODO assertion should be checked !
	// assert( !lagrange.isNull() );
	
	return vector(node->mechanicalState, lagrange.getId(node->mechanicalState) );	
}





// this is called on the top-down traversal, once for each node. we
// simply fetch infos for each dof.
void AssemblyVisitor::fill_prefix(simulation::Node* node) {
	assert( node->mechanicalState );
	assert( chunks.find( node->mechanicalState ) == chunks.end() );
	
	// fill chunk for current dof
	chunk& c = chunks[ node->mechanicalState ];
			
	c.size = node->mechanicalState->getMatrixSize();
	c.dofs = node->mechanicalState;
	
	vertex v; v.dofs = c.dofs; v.data = &c;
	
	c.M = mass( node );
	add(c.K, stiff( node ));
	
	if( !zero(c.M) || !zero(c.K) ) {
		c.mechanical = true;
		c.v = vel( node );
	}

	c.map = mapping( node );
	c.f = force( node );
	
	c.vertex = boost::add_vertex(v, graph);
	
	if( c.map.empty() ) {
		// independent
		// TODO this makes a lot of allocs :-/

		c.v = vel( node );
		c.f = force( node );
		c.P = proj( node );
		
	} else {
		// mapped
		c.C = compliance( node );
		
		if( !empty(c.C) ) {
			c.mechanical = true;
			
			c.phi = phi( node );
			c.damping = damping( node );

			// TODO this test should work but it doesn't :-/
			// if( !lagrange.isNull() ) {
			c.lambda = lambda( node );
			assert( c.lambda.size() );
			// } else {
			// 	std::cerr << "bad lagrange state vector :-/" << std::endl;
			// }
		}

	}

	
}



// bottom-up
void AssemblyVisitor::fill_postfix(simulation::Node* node) {
	assert( node->mechanicalState );
	assert( chunks.find( node->mechanicalState ) != chunks.end() );
	
	// fill chunk for current dof
	chunk& c = chunks[ node->mechanicalState ];

	for(chunk::map_type::const_iterator it = c.map.begin(), end = c.map.end(); 
	    it != end; ++it) {
				
		chunk& p = chunks[it->first];
		
		edge e;
		e.data = &it->second;
			
		// the edge is child -> parent
		boost::add_edge(c.vertex, p.vertex, e, graph); 
	}
	
}




void AssemblyVisitor::chunk::debug() const {
	using namespace std;
			
	cout << "chunk: " << dofs->getName() << endl
	     << "offset:" << offset << endl
	     << "size: " << size << endl
	     << "M:" << endl << M << endl
	     << "K:" << endl << K << endl
	     << "P:" << endl << P << endl
	     << "C:" << endl << C << endl
	     << "f:  " << f.transpose() << endl
	     << "v:  " << v.transpose() << endl
	     << "phi: " << phi.transpose() << endl
	     << "damping: " << damping << endl
	     << "map: " << endl
		;
	
	for(map_type::const_iterator mi = map.begin(), me = map.end();
	    mi != me; ++mi) {
		cout << "from: " << mi->first->getName() << endl
		     << "J: " << endl
		     << mi->second.J << endl
		     << "K: " << endl
		     << mi->second.K << endl
			;
	}
}
			
void AssemblyVisitor::debug() const {

	for(chunks_type::const_iterator i = chunks.begin(), e = chunks.end(); i != e; ++i ) {
		i->second.debug();			
	}
	
}


// TODO copypasta !!
void AssemblyVisitor::distribute_master(core::VecId id, const vec& data) {
	// scoped::timer step("solution distribution");
			
	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		const chunk* c = graph[prefix[i]].data; // find(chunks, prefix[i]);
		
		if( c->master() ) {
			vector(graph[prefix[i]].dofs, id, data.segment(off, c->size) );
			off += c->size;
		}
		
	}

	assert( data.size() == off );
}

// TODO copypasta !!
void AssemblyVisitor::distribute_compliant(core::VecId id, const vec& data) {
	// scoped::timer step("solution distribution");
			
	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
				
		const chunk* c = graph[prefix[i]].data; // find(chunks, prefix[i]);

		if( c->compliant() ) {
			vector(graph[prefix[i]].dofs, id, data.segment(off, c->size) );
			off += c->size;
		}
		
	}
	
	assert( data.size() == off );
}

// TODO copypasta
void AssemblyVisitor::distribute_compliant(core::behavior::MultiVecDeriv::MyMultiVecId id, const vec& data) {
	// scoped::timer step("solution distribution");
			
	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
		
		const chunk* c = graph[prefix[i]].data; // find(chunks, prefix[i]);
		
		if( c->compliant() ) { 
			vector( graph[ prefix[i] ].dofs, id.getId( graph[prefix[i]].dofs), data.segment(off, c->size) );
			off += c->size;
		}
		
	}
	
	assert( data.size() == off );
}



 
// this is used to propagate mechanical flag upwards mappings (prefix
// in the graph order)
struct AssemblyVisitor::propagation_helper {

	graph_type& g;
	
	propagation_helper(graph_type& g) : g(g) {} 
	
	void operator()( unsigned v ) const {
		
//		dofs_type* dofs = g[v].dofs;
		chunk* c = g[v].data;
		
		if( c->mechanical ) {
		
			for(graph_type::out_edge_range e = boost::out_edges(v, g); e.first != e.second; ++e.first) {
				
				chunk* p = g[ boost::target(*e.first, g) ].data; 
				p->mechanical = true;
				
				if(!zero( g[*e.first].data->K)) { 
					add(p->K, g[*e.first].data->K );
				}
			}
			
		}
	}

};



// multiplies mapping matrices together for everyone in the graph
struct AssemblyVisitor::process_helper {
	
	process_type& res;
	const graph_type& g;
	
	process_helper(process_type& res, const graph_type& g)
		: res(res), g(g)  {

	}

	void operator()(unsigned v) const {
		dofs_type* curr = g[v].dofs;
		chunk* c = g[v].data;
		
		const unsigned& size_m = res.size_m;
		full_type& full = res.full;
		offset_type& offsets = res.offset.master;
		
		if( !c->mechanical ) return;
		
		mat& Jc = full[ curr ];
		assert( empty(Jc) );
		
		// TODO use graph and out_edges
		for( graph_type::out_edge_range e = boost::out_edges(v, g); e.first != e.second; ++e.first) {
			
			vertex vp = g[ boost::target(*e.first, g) ];

			// parent data chunk/mapping matrix
			const chunk* p = vp.data; 
			mat& Jp = full[ vp.dofs ];
			{
				// mapping blocks
				const mat& jc = g[*e.first].data->J;
				
				// parent is not mapped: we put a shift matrix with the
				// correct offset as its full mapping matrix, so that its
				// children will get the right place on multiplication
				if( p->master() && empty(Jp) ) {
					// scoped::timer step("shift matrix");
					Jp = shift_right( find(offsets, vp.dofs), p->size, size_m);
				}
				
				// Jp is empty for children of a non-master dof (e.g. mouse)
				if(!empty(Jp) ){
					// scoped::timer step("mapping matrix product");
					
					// TODO optimize this, it is the most costly part
					add(Jc, jc * Jp );
				} else {
					assert( false && "parent has empty J matrix :-/" );
				}
			}
			
			if( ! (c->master() || !zero(Jc) )  )  {
				using namespace std;
				
				cerr << "houston we have a problem with " << c->dofs->getName()  << " under " << c->dofs->getContext()->getName() << endl
				     << "master: " << c->master() << endl 
				     << "mapped: " << (c->map.empty() ? string("nope") : p->dofs->getName() )<< endl
				     << "p mechanical ? " << p->mechanical << endl
				     << "empty Jp " << empty(Jp) << endl
				     << "empty Jc " << empty(Jc) << endl;
				
				assert( false );
			}
			
		
			
		}



	}; 


};

struct AssemblyVisitor::prefix_helper {
	prefix_type& res;

	prefix_helper(prefix_type& res) : res(res) {
		res.clear();
	}
	
	template<class G>
	void operator()(unsigned v, const G& ) const {
		res.push_back( v );
	}
	
};



AssemblyVisitor::process_type AssemblyVisitor::process() const {
	// scoped::timer step("mapping processing");
	
	process_type res;

	unsigned& size_m = res.size_m;
	unsigned& size_c = res.size_c;

//	full_type& full = res.full;
			
	// independent dofs offsets (used for shifting parent)
	offset_type& offsets = res.offset.master;
			
	unsigned off_m = 0;
	unsigned off_c = 0;
	
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
		
		const chunk* c = graph[ prefix[i] ].data;
		
		// independent
		if( c->master() ) {
			offsets[ graph[ prefix[i] ].dofs ] = off_m;
			off_m += c->size;
		} else if( !empty(c->C) ) {
			off_c += c->size;
		}
		
	}
	
	// update total sizes
	size_m = off_m;
	size_c = off_c;
	
	// prefix mapping concatenation and stuff 
	std::for_each(prefix.begin(), prefix.end(), process_helper(res, graph) ); 	// TODO merge with offsets computation ?

	return res;
}




// this is meant to optimize L^T D L products
static inline AssemblyVisitor::mat ltdl(const AssemblyVisitor::mat& l, 
                                        const AssemblyVisitor::mat& d) {
	return l.transpose() * (d * l);
}
		

// produce actual system assembly
AssemblyVisitor::system_type AssemblyVisitor::assemble() const{
	assert(!chunks.empty() && "need to send a visitor first");

	// concatenate mappings and obtain sizes
	process_type p = process();
			
	// result system
	system_type res(p.size_m, p.size_c);
	
	res.dt = mparams->dt();
			
	// master/compliant offsets
	unsigned off_m = 0;
	unsigned off_c = 0;
			
	SReal dt2 = res.dt * res.dt;

	// assemble system
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		// current chunk
		const chunk* c = graph[ prefix[i] ].data;
		assert( c->size );

		if( !c->mechanical ) continue;
		
		// full mapping chunk
		const mat& Jc = p.full[ graph[ prefix[i] ].dofs ];
		
		// independent dofs: fill mass/stiffness
		if( c->master() ) {
			res.master.push_back( c->dofs );

			// scoped::timer step("independent dofs");
			mat shift = shift_right(off_m, c->size, p.size_m);
			
			mat H(c->size, c->size);

			// mass matrix / momentum
			if( !zero(c->M) ) {
				
				H += c->M; 
				res.p.segment(off_m, c->size).noalias() += c->M * c->v;
			}
			
			// stiffness matrix
			if( !zero(c->K) )  {
				// res.K.middleRows(off_m, c.size) = Kc * shift;
				H -= dt2 * c->K;
				
				// res.H.middleRows(off_m, c.size) = res.H.middleRows(off_m, c.size) - dt2 * (Kc * shift);
			}
			
			// TODO this is costly, find a faster way to do it
			if( !zero(H) ) {
				res.H.middleRows(off_m, c->size) = res.H.middleRows(off_m, c->size) + H * shift;
			}
			
			// these should not be empty anyways
			if( !zero(c->f) ) res.f.segment(off_m, c->size) = c->f;
			if( !zero(c->v) ) res.v.segment(off_m, c->size) = c->v;
			if( !zero(c->P) ) res.P.middleRows(off_m, c->size) = c->P * shift;
			
			off_m += c->size;
		} 
				
		// mapped dofs
		else { 

			if( !zero(Jc) ) {
				assert( Jc.cols() == int(p.size_m) );
				
				{
					mat H(c->size, c->size);
					
					if( !zero(c->M) ) {
						// contribute mapped mass
						assert( c->v.size() == int(c->size) );
						assert( c->M.cols() == int(c->size) ); 
						assert( c->M.rows() == int(c->size) );

						// momentum
						res.p.noalias() += Jc.transpose() * (c->M * c->v);

						H += c->M;
					}
					
					// mapped stiffness
					if( !zero(c->K) ) {
						H -= dt2 * c->K;
					}
					
					// actual response matrix mapping
					if( !zero(H) ) {
						res.H += ltdl(Jc, H);
					}
					
				}
				
			}					
					
			// compliant dofs: fill compliance/phi/lambda
			if( c->compliant() ) { 
				res.compliant.push_back( c->dofs );
				// scoped::timer step("compliant dofs");
				assert( !zero(Jc) );
				
				// mapping
				res.J.middleRows(off_c, c->size) = Jc;
						
				// compliance
				if(!zero(c->C) ) {
					
					// you never know
					assert( mparams->implicitVelocity() == 1 );
					assert( mparams->implicitPosition() == 1 );
					
					SReal l = res.dt + c->damping;

					// zero damping => C / dt2

					SReal factor = 1.0 / (res.dt * l );
					res.C.middleRows(off_c, c->size) = 
						c->C * shift_right(off_c, c->size, p.size_c, factor);
				}
			
				// phi
				res.phi.segment(off_c, c->size) = c->phi;
				
				if( c->lambda.size() ) res.lambda.segment( off_c, c->size ) = c->lambda;
				
				off_c += c->size;
			}
		}
	}

	assert( off_m == p.size_m );
	assert( off_c == p.size_c );

	return res;
}
		

bool AssemblyVisitor::chunk::check() const {

	// let's be paranoid
	assert( dofs && size == dofs->getMatrixSize() );

	if( master() ) {
		assert( empty(C) );
		assert( !empty(P) );
		
		assert( f.size() == int(size) );
		assert( v.size() == int(size) );
		
	} else {
		
		if(!empty(C)) {
			assert( phi.size() == int(size) );
			assert( damping >= 0 );
		}

	}

	// should be outer size ?
	assert( empty(K) || K.rows() == int(size) );
	assert( empty(M) || M.rows() == int(size) );

	return true;
}

		
void AssemblyVisitor::clear() {

	chunks.clear();
	prefix.clear();
	graph.clear();
	
}


		
Visitor::Result AssemblyVisitor::processNodeTopDown(simulation::Node* node) {
	if( !start_node ) start_node = node;
	
	if( node->mechanicalState ) fill_prefix( node );
	return RESULT_CONTINUE;
}

void AssemblyVisitor::processNodeBottomUp(simulation::Node* node) {
	if( node->mechanicalState ) fill_postfix( node );

	// are we finished yo ?
	if( node == start_node ) {

		// backup prefix traversal order
		utils::dfs( graph, prefix_helper( prefix ) );
		
		// postfix mechanical flags propagation (and geometric stiffness matrices)
		std::for_each(prefix.rbegin(), prefix.rend(), propagation_helper(graph) );
		
		// TODO at this point it could be a good thing to prune
		// non-mechanical nodes in the graph, in order to avoid unneeded
		// mapping concatenations, then rebuild the prefix order
		
		start_node = 0;
	}
}
		


}
}

