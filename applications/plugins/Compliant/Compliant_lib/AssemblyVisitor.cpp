#include "AssemblyVisitor.h"

#include <sofa/component/linearsolver/EigenVectorWrapper.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>

#include "SolverFlags.h"
#include "Projector.h"

#include "./utils/scoped.h"
#include "./utils/find.h"
#include "./utils/cast.h"
#include "./utils/sparse.h"

#include <boost/graph/graphviz.hpp>

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


		
// right-shift, size x (off + size) matrix: (0, id)
static AssemblyVisitor::mat shift_right(unsigned off, unsigned size, unsigned total_cols) {
	AssemblyVisitor::mat res( size, total_cols); 
	assert( total_cols >= (off + size) );
	
	// res.resize(size, total_cols );
	// res.reserve(Eigen::VectorXi::Constant(size, 1));
	res.reserve( size );
	
	for(unsigned i = 0; i < size; ++i) {
		res.startVec( i );
		res.insertBack(i, off + i) = 1.0;
		// res.insert(i, off + i) = 1.0;
	}
	res.finalize();
	
	// res.makeCompressed(); // TODO is this needed ?
	return res;
}


		
		
AssemblyVisitor::AssemblyVisitor(const core::MechanicalParams* mparams) 
	: base( mparams ),
	  mparams( mparams )

{ }
		
AssemblyVisitor::chunk::chunk() 
	: offset(0), 
	  size(0), 
	  damping(0), 
	  mechanical(false), 
	  vertex(-1), 
	  dofs(0) {
	
}

		
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
		
		
void AssemblyVisitor::vector(dofs_type* dofs, core::VecId id, const vec::ConstSegmentReturnType& data) {
	volatile unsigned int size = dofs->getMatrixSize();
	assert( size == data.size() );
			
	// realloc
	if( data.size() > tmp.size() ) tmp.resize(data.size() );
	tmp.head(data.size()) = data;
	
	unsigned off = 0;
	wrap w(tmp);
	dofs->copyFromBaseVector(id, &w, off);
	assert( off == data.size() );
}		
		

AssemblyVisitor::vec AssemblyVisitor::vector(simulation::Node* node, core::ConstVecId id) {
	assert( node->mechanicalState );
				
	unsigned size = node->mechanicalState->getMatrixSize();

	const bool fast = false;
			
	if( fast ) {
		// map data
		const void* data = node->mechanicalState->baseRead(id)->getValueVoidPtr();
			
		// TODO need to know if we're dealing with double or floats
		return Eigen::Map<const vec>( reinterpret_cast<const double*>(data), size);
	} else {
				
		// realloc
		if( size > tmp.size() ) tmp.resize(size);

		unsigned off = 0;
		wrap w(tmp);
				
		node->mechanicalState->copyToBaseVector(&w , id, off);
				
		return tmp.head(size);
	}
}


// pretty prints a mapping
static inline std::string mapping_name(simulation::Node* node) {
	return node->mechanicalMapping->getName() + " (class: " + node->mechanicalMapping->getClassName() + ") ";
}
		
		
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
	unsigned rows = to->getMatrixSize();
	
	for( unsigned i = 0, n = from.size(); i < n; ++i ) {

		// parent dofs
		dofs_type* p = safe_cast<dofs_type>(from[i]);  
		
		// skip non-mechanical dofs
		if(!p) continue;
				
		// mapping wrt p
		chunk::mapped& c = res[p];

		if( js ) c.J = convert( (*js)[i] );
				
		if( empty(c.J) ) {
			unsigned cols = p->getMatrixSize(); 
			
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


				
// TODO optimize !!!
struct derp_matrix : defaulttype::BaseMatrix {
	static void derp() { throw std::logic_error("not implemented"); }

	unsigned rowSize() const { derp(); return 0; }
	unsigned colSize() const { derp(); return 0; }

	SReal element(int , int ) const { derp(); return 0; }
	void resize(int , int ) { derp(); }
	void clear() { derp(); }

	void set(int, int, double) { derp(); }
	void add(int, int, double) { derp(); }
};

struct add_matrix : derp_matrix {

	AssemblyVisitor::mat& res;
	
	typedef std::map< std::pair<int, int>, SReal > values_type;
	values_type values;
	
	add_matrix(AssemblyVisitor::mat& res) : res(res) { 
		// res should have nnz allocated first
		last.i = -1; last.j = -1;
	}

	~add_matrix() { 
		res.reserve( values.size() );
		
		int row = -1;
		for(values_type::const_iterator it = values.begin(), end = values.end(); 
		    it != end; ++it) {
			int i = it->first.first;
			int j = it->first.second;
			
			if( row == -1 || row < i ) res.startVec(i);
			res.insertBack(i, j) = it->second;
		}
		
		
		res.finalize();
	}
	
	unsigned rowSize() const { return res.rows(); }
	unsigned colSize() const { return res.cols(); }
	
	struct { int i, j; } last;
	
	void add(int i, int j, double value) {

		if( value ) values[ std::make_pair(i, j) ] += value;
		
		// if( last.i == -1 || i > last.i ) res.startVec( i );
		// res.insertBack(i, j) = value;
 
		// last.i = i;
		// last.j = j;
	}

};
 

AssemblyVisitor::mat AssemblyVisitor::mass(simulation::Node* node) {
	unsigned size = node->mechanicalState->getMatrixSize();

	mat res(size, size);
	
	if( !node->mass ) return res;
	
	typedef EigenBaseSparseMatrix<SReal> Sqmat;
	Sqmat sqmat( size, size );

	{
		// add_matrix add(res);
		// SingleMatrixAccessor accessor( &add );

		SingleMatrixAccessor accessor( &sqmat );
		node->mass->addMToMatrix( mparams, &accessor );
	}
 	
	sqmat.compress();
	res = sqmat.compressedMatrix.selfadjointView<Eigen::Upper>();
	return res;
}
		

		
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

		
AssemblyVisitor::mat AssemblyVisitor::compliance(simulation::Node* node) {
			
	mat res;
			
	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
					
		const BaseMatrix* c = ffield->getComplianceMatrix(mparams);
				
		if( c ) return convert( c );
				
	}
			
	return res;
}

		
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

		
AssemblyVisitor::vec AssemblyVisitor::force(simulation::Node* node) {
	assert( node->mechanicalState );
	return vector(node, core::VecDerivId::force());	
}

AssemblyVisitor::vec AssemblyVisitor::vel(simulation::Node* node) {
	assert( node->mechanicalState );
	return vector(node, core::VecDerivId::velocity());	
}

AssemblyVisitor::real AssemblyVisitor::damping(simulation::Node* node) {
	assert( node->mechanicalState );
			
	assert( node->forceField.size() <= 1 );

	for(unsigned i = 0; i < node->forceField.size(); ++i ) {
		BaseForceField* ffield = node->forceField[i];
		return ffield->getDampingRatio();
	}
	
	return 0;
}

AssemblyVisitor::vec AssemblyVisitor::rhs(simulation::Node* node) {
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

AssemblyVisitor::chunk::extra_type AssemblyVisitor::extra(simulation::Node* node) { 
	// hacks !

	assert( node->mechanicalState );
	
	// look for a projector
	return 
		node->mechanicalState->getContext()->get<component::linearsolver::Projector>(core::objectmodel::BaseContext::Local);
	
}

AssemblyVisitor::system_type::flags_type AssemblyVisitor::flags(simulation::Node* node) {
	assert( node->mechanicalState );
	
	component::linearsolver::SolverFlags* flags = 
		node->mechanicalState->getContext()->get<component::linearsolver::SolverFlags>(core::objectmodel::BaseContext::Local);
	
	system_type::flags_type res;
	
	if( flags ) {
		unsigned n = node->mechanicalState->getMatrixSize();
		res.resize( n );
		unsigned written = flags->write( res.data() );
		assert( written == n );
		
	}
	
	return res;
}
	

// top-down
void AssemblyVisitor::fill_prefix(simulation::Node* node) {
	assert( node->mechanicalState );
	assert( chunks.find( node->mechanicalState ) == chunks.end() );
	
	// std::cerr << "prefix fill " << pretty(node->mechanicalState) << std::endl;


	// backup prefix order
	prefix.push_back( node->mechanicalState );

	// fill chunk for current dof
	chunk& c = chunks[ node->mechanicalState ];
			
	c.size = node->mechanicalState->getMatrixSize();
	c.dofs = node->mechanicalState;
	
	vertex v; v.dofs = c.dofs;
	
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
			c.phi = rhs( node );
			c.damping = damping( node );
			c.flags = flags( node );

			// hack
			c.extra = extra( node );
			
			c.mechanical = true;
		
		}

	}

	// if( c.mechanical ) std::cerr << "mechanincal !" << std::endl;
	
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




void AssemblyVisitor::debug_chunk(const AssemblyVisitor::chunks_type::const_iterator& i) {
	using namespace std;
			
	cout << "chunk: " << i->first->getName() << endl
	     << "offset:" << i->second.offset << endl
	     << "size: " << i->second.size << endl
	     << "M:" << endl << i->second.M << endl
	     << "K:" << endl << i->second.K << endl
	     << "P:" << endl << i->second.P << endl
	     << "C:" << endl << i->second.C << endl
	     << "f:  " << i->second.f.transpose() << endl
	     << "v:  " << i->second.v.transpose() << endl
	     << "phi: " << i->second.phi.transpose() << endl
	     << "damping: " << i->second.damping << endl
	     << "map: " << endl
		;
	for(chunk::map_type::const_iterator mi = i->second.map.begin(), me = i->second.map.end();
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
		debug_chunk(i);			
	}
			
}


// TODO copypasta !!
void AssemblyVisitor::distribute_master(core::VecId id, const vec& data) {
	// scoped::timer step("solution distribution");
			
	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
				
		// TODO optimize map lookup by saving prefix independent state
		// list 
		// TODO or: save offsets in chunks ?
		const chunk& c = find(chunks, prefix[i]);

		// paranoia
		// c.check();
		
		if( c.master() ) {
			vector(prefix[i], id, data.segment(off, c.size) );
			off += c.size;
		}
		
	}

	assert( data.size() == off );
}

// TODO copypasta !!
void AssemblyVisitor::distribute_compliant(core::VecId id, const vec& data) {
	// scoped::timer step("solution distribution");
			
	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
				
		// TODO optimize map lookup by saving prefix independent state
		// list 
		// TODO or: save offsets in chunks ?
		const chunk& c = find(chunks, prefix[i]);

		// paranoia
		// c.check();
		
		if( c.compliant() ) {
			vector(prefix[i], id, data.segment(off, c.size) );
			off += c.size;
		}
		
	}
	
	assert( data.size() == off );
}

 
// this is used to propagate mechanical flag upwards mappings (prefix
// in the graph order)
struct AssemblyVisitor::propagation_helper {

	// should be const lol
	chunks_type& chunks;

	propagation_helper(chunks_type& chunks) : chunks(chunks) {} 
	
	void operator()( dofs_type* dofs) const {
		// TODO use out_edges instead lol ?
		chunk& c = chunks[ dofs ];
		
		// std::cerr << "propagating from " << pretty(c.dofs) << std::endl;

		if( c.mechanical ) {
			
			for(chunk::map_type::const_iterator mi = c.map.begin(), me = c.map.end();
			    mi != me; ++mi) {

				chunk& p = chunks[ mi->first ];
				p.mechanical = true;
							
				if(!zero(mi->second.K)) { 
					add(p.K, mi->second.K);
				}
				// std::cerr << "becomes mechanical lol " << pretty(mi->first) << std::endl;
			}
		
		}
	}

	template<class G>
	void operator()(unsigned v, const G& g) const {
		(*this)(g[v].dofs);
	}
	
};



template<class G>
struct writer {
	const G& g;
	writer(const G& g) :g(g) { }
	
	void operator()(std::ostream& out, typename G::vertex_descriptor vertex) const {
		out << "[label=\"" << g[vertex].dofs->getContext()->getName() << "\"]";
	}

	void operator()(std::ostream& out, typename G::edge_descriptor edge) const {
		// out << g[vertex].dofs->getContext()->getName();
	}
	
	
};

template<class G>
static void debug_graph(const G& g) {
	std::ofstream out("/tmp/graph");
	boost::write_graphviz(out, g, writer<G>(g));
}


struct AssemblyVisitor::process_helper {
	
	process_type& res;
	const chunks_type& chunks;
	
	process_helper(process_type& res,
	               const chunks_type& chunks)
		: res(res),
		  chunks(chunks) {
		// std::cerr << "avanti" << std::endl;
	}

	template<class G>
	void operator()(unsigned v, const G& g) const {
		(*this)(g[v].dofs);
	}

	void operator()(dofs_type* curr) const {
		// cerr << "processing " << pretty( curr ) << std::endl;

		const unsigned& size_m = res.size_m;
		full_type& full = res.full;
		offset_type& offsets = res.offset.master;
		
		// current data chunk/mapping matrix
		const chunk& c = find(chunks, curr);

		if( !c.mechanical ) return;
		
		mat& Jc = full[ curr ];
		assert( empty(Jc) );
		
		// add regular stiffness 
		// mat& Kc = fc.K;
		// add(Kc, c.K);
		
		// concatenate mapping wrt independent dofs
		for(chunk::map_type::const_iterator mi = c.map.begin(), me = c.map.end();
		    mi != me; ++mi) {

			// parent data chunk/mapping matrix
			const chunk& p = find(chunks, mi->first);
			mat& Jp = full[ mi->first ];
			{
				// scoped::timer step("mapping concatenation");
				
				// mapping blocks
				const mat& jc = mi->second.J;
				
				// parent is not mapped: we put a shift matrix with the
				// correct offset as its full mapping matrix, so that its
				// children will get the right place on multiplication
				if( p.master() && empty(Jp) ) {
					// scoped::timer step("shift matrix");
					Jp = shift_right( find(offsets, mi->first), p.size, size_m);
				}
				
				// Jp is empty for children of a non-master dof (e.g. mouse)
				if(!empty(Jp) ){
					// scoped::timer step("mapping matrix product");
					
					// TODO optimize this, it is the most costly part
					add(Jc, jc * Jp );
				} else {
					std::cerr << "parent: " << pretty(p.dofs) << " has empty J matrix" << std::endl;
					assert( false );
				}
			}
 
			if( ! (c.master() || !zero(Jc) )  )  {
				using namespace std;

				cerr << "houston we have a problem with " << c.dofs->getName()  << " under " << c.dofs->getContext()->getName() << endl
				     << "master: " << c.master() << endl 
				     << "mapped: " << (c.map.empty() ? string("nope") : p.dofs->getName() )<< endl
				     << "p mechanical ? " << p.mechanical << endl
				     << "empty Jp " << empty(Jp) << endl
				     << "empty Jc " << empty(Jc) << endl;
				
				assert( false );
				
				// { 
				// 	// scoped::timer step("geometric stiffness propagation");
				// 	// geometric stiffness
				// 	const mat& kc = mi->second.K;

				// 	if( !zero(kc) ) {
				// 		// std::cerr << "geometric stiffness lol" << std::endl;
				// 		// contribute stiffness block to parent
				// 		mat& Kp = fp.K;
				// 		add(Kp, kc);
				// 	}
				// }

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
	void operator()(unsigned v, const G& g) const {
		res.push_back(g[v].dofs);
	}
	
};



AssemblyVisitor::process_type AssemblyVisitor::process() const {
	// scoped::timer step("mapping processing");


	process_type res;

	unsigned& size_m = res.size_m;
	unsigned& size_c = res.size_c;

	full_type& full = res.full;
			
	// independent dofs offsets (used for shifting parent)
	offset_type& offsets = res.offset.master;
			
	unsigned off_m = 0;
	unsigned off_c = 0;
	
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
		
		const chunk& c = find(chunks, prefix[i]);
		// independent
		if( c.master() ) {
			offsets[ prefix[i] ] = off_m;
			off_m += c.size;
		} else if( !empty(c.C) ) {
			off_c += c.size;
		}
				
	}
	
	// update total sizes
	size_m = off_m;
	size_c = off_c;
	
	// postfix mechanical flags propagation
	std::for_each(prefix.rbegin(), prefix.rend(), propagation_helper(chunks) );

	// prefix mapping propagation TODO merge with offsets computation
	std::for_each(prefix.begin(), prefix.end(), process_helper(res, chunks) );
	
	return res;
}


template<class It1, class It2, class F>
static inline void parallel_iter(It1& it1,
                                 It2& it2,
                                 const F& f) {
	while( it1 && it2 ) {
		if( it1.col() < it2.row() ) ++it1;
		else if( it1.col() > it2.row() ) ++it2;
		else {
			f(it1.value(), it2.value());
			++it1, ++it2;
		}
	}
	
}

struct add_prod {
	SReal& res;
	
	add_prod(SReal& res) : res(res) { }
	
	inline void operator()(const SReal& x, const SReal& y) const {
		res += x * y;
	}
	
};



struct dot {
	const AssemblyVisitor::mat& l;
	const AssemblyVisitor::mat& d;
	
	dot(const AssemblyVisitor::mat& l,
	    const AssemblyVisitor::mat& d) 
		: l(l), d(d) { }
	

	inline SReal operator()(unsigned i, unsigned j) const {
		SReal res = 0;
	
		for(unsigned u = 0, n = d.rows(); u < n; ++u) {
			AssemblyVisitor::mat::InnerIterator duv(d, u);
		
			unsigned v = duv.col();
		
			res += l.coeff(u, i) * duv.value() * l.coeff(v, j);
		}

		
		// for(AssemblyVisitor::cmat::InnerIterator lui(l, i); lui; ++lui) {

		// 	SReal tmp = 0;
		// 	AssemblyVisitor::mat::InnerIterator duv(d, lui.row() );
		// 	AssemblyVisitor::cmat::InnerIterator lvj(l, j);

		// 	parallel_iter(duv, lvj, add_prod(tmp) );
			
		// 	res += lui.value() * tmp;
		// }
		
		return res;
	}

};

// this is meant to optimize L^T D L products
static inline AssemblyVisitor::mat ltdl(const AssemblyVisitor::mat& l, 
                                        const AssemblyVisitor::mat& d) {
	return l.transpose() * (d * l);
}
		

// TODO organize this mess
AssemblyVisitor::system_type AssemblyVisitor::assemble() const{
	// scoped::timer step("system assembly");
	assert(!chunks.empty() && "need to send a visitor first");

	// concatenate mappings and obtain sizes
	process_type p = process();
			
	// result system
	system_type res(p.size_m, p.size_c);
	
	// TODO tighter
	unsigned n_blocks = prefix.size();
	res.blocks.reserve( n_blocks );
	
	res.dt = mparams->dt();
			
	// master/compliant offsets
	unsigned off_m = 0;
	unsigned off_c = 0;
			
	SReal dt2 = res.dt * res.dt;

	// assemble system
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		// current chunk
		const chunk& c = find(chunks, prefix[i]);
		assert( c.size );

		if( !c.mechanical ) continue;
		
		// full mapping chunk
		const mat& Jc = p.full[ prefix[i] ];
		
		// independent dofs: fill mass/stiffness
		if( c.master() ) {
			// scoped::timer step("independent dofs");
			mat shift = shift_right(off_m, c.size, p.size_m);
				
			mat H(c.size, c.size);
			
			if( !zero(c.M) ) {
				
				// res.M.middleRows(off_m, c.size) = c.M * shift;
				H += c.M; // res.H.middleRows(off_m, c.size) = res.H.middleRows(off_m, c.size) + c.M * shift;
				
				res.p.segment(off_m, c.size).noalias() += c.M * c.v;
			}
			
			if( !zero(c.K) )  {
				// res.K.middleRows(off_m, c.size) = Kc * shift;
				H -= dt2 * c.K;
				
				// res.H.middleRows(off_m, c.size) = res.H.middleRows(off_m, c.size) - dt2 * (Kc * shift);
			}
			
			if( !zero(H) ) {
				res.H.middleRows(off_m, c.size) = res.H.middleRows(off_m, c.size) + H * shift;
			}
			
			// these should not be empty
			if( !zero(c.f) ) res.f.segment(off_m, c.size) = c.f;
			if( !zero(c.v) ) res.v.segment(off_m, c.size) = c.v;
			if( !zero(c.P) ) res.P.middleRows(off_m, c.size) = c.P * shift;
			
			off_m += c.size;
		} 
				
		// mapped dofs
		else { 

			if( !zero(Jc) ) {
				assert( Jc.cols() == int(p.size_m) );
				
				// scoped::timer step("mass/stiffness mapping");
						
				// TODO possibly merge these two operations ?
// #pragma omp parallel sections
				{
// #pragma omp section
					mat H(c.size, c.size);
					
					if( !zero(c.M) ) {
						// contribute mapped mass
						// res.M += ltdl(Jc, c.M); 

						assert( c.v.size() == int(c.size) );
						assert( c.M.cols() == int(c.size) ); 
						assert( c.M.rows() == int(c.size) );

						// momentum
						res.p.noalias() += Jc.transpose() * (c.M * c.v);

						H += c.M;
					}
// #pragma omp section

					if( !zero(c.K) ) {
						// res.K += ltdl(Jc, Kc); // contribute mapped stiffness
						
						H -= dt2 * c.K;
					}
					
					if( !zero(H) ) {
						res.H += ltdl(Jc, H);
					}
					
				}
				
			}					
					
			// compliant dofs: fill compliance matrix/rhs
			if( c.compliant() ) { // !empty(c.C)
				// scoped::timer step("compliant dofs");
				assert( !zero(Jc) );
				
				// mapping
				res.J.middleRows(off_c, c.size) = Jc;
						
				// compliance
				if(!zero(c.C) ) {
					
					// you never know
					assert( mparams->implicitVelocity() == 1 );
					assert( mparams->implicitPosition() == 1 );

					SReal l = res.dt + c.damping;

					// zero damping => C / dt2
					res.C.middleRows(off_c, c.size) = ( c.C / (res.dt * l )) * shift_right(off_c, c.size, p.size_c);
				}
			
				// rhs
				res.phi.segment(off_c, c.size) = c.phi;
				
				// damping 
				// res.damping.segment(off_c, c.size).setConstant( c.damping );
				
				// equation flags
				if( c.flags.size() ) {
					res.flags.segment(off_c, c.size) = c.flags;
				}
				
				{
					// add blocks, one per compliant dof
					unsigned off = off_c;
					unsigned dim = c.dofs->getDerivDimension();
				
					for( unsigned k = 0, max = c.dofs->getSize(); k < max; ++k) {
					
						AssembledSystem::block block(off, dim);
					
						// hack
						block.data = c.extra;
					
						res.blocks.push_back( block );
						off += dim;
					}
					assert( off == off_c + c.size );
				}
				
				off_c += c.size;
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
			assert( !flags.size() || flags.size() == int(size) );
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
	if( node->mechanicalState ) fill_prefix( node );
	return RESULT_CONTINUE;
}

void AssemblyVisitor::processNodeBottomUp(simulation::Node* node) {
	if( node->mechanicalState ) fill_postfix( node );

	// are we finished ?
	if( node->getParents().empty() ) utils::dfs( graph, prefix_helper( prefix ) );
	
}
		

}
}

