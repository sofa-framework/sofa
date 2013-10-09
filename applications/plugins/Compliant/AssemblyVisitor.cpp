#include "AssemblyVisitor.h"


#include <sofa/component/linearsolver/EigenVectorWrapper.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>

#include "./utils/scoped.h"
#include "./utils/cast.h"
#include "./utils/sparse.h"


namespace sofa {
namespace simulation {

using namespace component::linearsolver;
using namespace core::behavior;

typedef EigenVectorWrapper<SReal> wrap;









AssemblyVisitor::AssemblyVisitor(const core::MechanicalParams* mparams, const core::MechanicalParams* mparamsWithoutStiffness, MultiVecDerivId velId, MultiVecDerivId lagrangeId)
	: base( mparams ),
      mparams( mparams ),
      mparamsWithoutStiffness( mparamsWithoutStiffness ),
      _velId(velId),
      lagrange(lagrangeId),
      start_node(0),
      _processed(0)
{
}


AssemblyVisitor::~AssemblyVisitor()
{
    if( _processed ) delete _processed;
}


AssemblyVisitor::chunk::chunk()
	: offset(0),
	  size(0),
	  damping(0),
	  mechanical(false),
	  vertex(-1),
	  dofs(0) {

}

// this is not thread safe lol
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

//	dofs_type* to = safe_cast<dofs_type>(node->mechanicalMapping->getTo()[0]);
//	unsigned rows = to->getMatrixSize();

	for( unsigned i = 0, n = from.size(); i < n; ++i ) {

		// parent dofs
		dofs_type* p = safe_cast<dofs_type>(from[i]);

		// skip non-mechanical dofs
		if(!p) continue;

		// mapping wrt p
		chunk::mapped& c = res[p];

        if( js ) c.J = convert<mat>( (*js)[i] );

		if( empty(c.J) ) {
//			unsigned cols = p->getMatrixSize();

			std::string msg("empty mapping block for " + mapping_name(node) + " (is mapping matrix assembled ?)" );
			assert( false );

			throw std::logic_error(msg);
		}

        if( ks && (*ks)[i] ) {
            c.K = convert<mat>( (*ks)[i] );
        }

	}

	return res;
}






// projection matrix
AssemblyVisitor::mat AssemblyVisitor::proj(simulation::Node* node) {
	assert( node->mechanicalState );

	unsigned size = node->mechanicalState->getMatrixSize();

	// identity matrix TODO alloc ?
    tmp_p.compressedMatrix = shift_right<mat>(0, size, size);

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

        if( !ffield->isCompliance.getValue() ) continue;

		const BaseMatrix* c = ffield->getComplianceMatrix(mparams);

        if( c ) return convert<mat>( c );
#ifndef NDEBUG
		else std::cerr<<SOFA_CLASS_METHOD<<ffield->getName()<<" getComplianceMatrix not implemented"<< std::endl;
#endif

	}

    return res;
}


// ode matrix
AssemblyVisitor::mat AssemblyVisitor::odeMatrix(simulation::Node* node)
{
    unsigned size = node->mechanicalState->getMatrixSize();

    typedef EigenBaseSparseMatrix<SReal> Sqmat;
    Sqmat sqmat( size, size );

    // note that mass are included in forcefield
    for(unsigned i = 0; i < node->forceField.size(); ++i )
    {
        BaseForceField* ffield = node->forceField[i];

        SingleMatrixAccessor accessor( &sqmat );

        // when it is a compliant, you need to add M if mass, B and part of K corresponding to rayleigh damping
        ffield->addMBKToMatrix( ffield->isCompliance.getValue()?mparamsWithoutStiffness:mparams, &accessor );

    }

    sqmat.compress();
    return sqmat.compressedMatrix.selfadjointView<Eigen::Upper>();
}


// fetches force
AssemblyVisitor::vec AssemblyVisitor::force(simulation::Node* node) {
	assert( node->mechanicalState );
	return vector(node->mechanicalState, core::VecDerivId::force());
}


// fetches velocity
AssemblyVisitor::vec AssemblyVisitor::vel(simulation::Node* node, MultiVecDerivId velId ) {
	assert( node->mechanicalState );
//	return vector(node->mechanicalState, core::VecDerivId::velocity());

    return vector( node->mechanicalState, velId.getId(node->mechanicalState) );

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

        if( ffield->isCompliance.getValue() ) {
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

    c.H = odeMatrix( node );

    if( !zero(c.H) ) {
		c.mechanical = true;
        c.v = vel( node, _velId );
	}

    c.map = mapping( node );

	c.vertex = boost::add_vertex(graph);
	graph[c.vertex] = v;
	
	if( c.map.empty() ) {
		// independent
		// TODO this makes a lot of allocs :-/

		c.v = vel( node, _velId );
        c.b = force( node );
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
		graph_type::edge_descriptor ed = boost::add_edge(c.vertex, p.vertex, graph).first;
		graph[ed] = e;
	}

}




void AssemblyVisitor::chunk::debug() const {
	using namespace std;

	cout << "chunk: " << dofs->getName() << endl
	     << "offset:" << offset << endl
         << "size: " << size << endl
         << "H:" << endl << H << endl
	     << "P:" << endl << P << endl
	     << "C:" << endl << C << endl
         << "b:  " << b.transpose() << endl
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
void AssemblyVisitor::distribute_master( core::behavior::MultiVecDeriv::MyMultiVecId id, const vec& data) {
	// scoped::timer step("solution distribution");

	unsigned off = 0;
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		const chunk* c = graph[prefix[i]].data; // find(chunks, prefix[i]);

		if( c->master() ) {
            vector(graph[prefix[i]].dofs, id.getId( graph[prefix[i]].dofs), data.segment(off, c->size) );
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

    const core::MechanicalParams* mparams;
    graph_type& g;

    propagation_helper(const core::MechanicalParams* mparams, graph_type& g) : mparams(mparams), g(g) {}

    void operator()( unsigned v ) const {

//		dofs_type* dofs = g[v].dofs;
        chunk* c = g[v].data;

        if( c->mechanical ) {

            for(graph_type::out_edge_range e = boost::out_edges(v, g); e.first != e.second; ++e.first) {

                chunk* p = g[ boost::target(*e.first, g) ].data;
                p->mechanical = true;

                if(!zero( g[*e.first].data->K)) {
                    add(p->H, mparams->kFactor() * g[*e.first].data->K );
                }
            }

        }
    }

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



AssemblyVisitor::process_type* AssemblyVisitor::process() const {
	// scoped::timer step("mapping processing");

    process_type* res = new process_type();

    unsigned& size_m = res->size_m;
    unsigned& size_c = res->size_c;

//	full_type& full = res->full;

	// independent dofs offsets (used for shifting parent)
    offset_type& offsets = res->offset.master;

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
    std::for_each(prefix.begin(), prefix.end(), process_helper(*res, graph) ); 	// TODO merge with offsets computation ?

	return res;
}




// this is meant to optimize L^T D L products
static inline AssemblyVisitor::mat ltdl(const AssemblyVisitor::mat& l,
                                        const AssemblyVisitor::mat& d) {
	return l.transpose() * (d * l);
}


#define USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX 0

// produce actual system assembly
AssemblyVisitor::system_type AssemblyVisitor::assemble() const {
	assert(!chunks.empty() && "need to send a visitor first");

	// assert( !_processed );

	// concatenate mappings and obtain sizes
    _processed = process();

	// result system
    system_type res(_processed->size_m, _processed->size_c);

	res.dt = mparams->dt();

	// master/compliant offsets
	unsigned off_m = 0;
	unsigned off_c = 0;

#if USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX
    typedef Eigen::Triplet<real> Triplet;
    std::vector<Triplet> H_triplets, C_triplets, P_triplets;
#endif

	// assemble system
	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		// current chunk
        const chunk& c = *graph[ prefix[i] ].data;
        assert( c.size );

        if( !c.mechanical ) continue;

		// full mapping chunk
        const mat& Jc = _processed->full[ graph[ prefix[i] ].dofs ];

		// independent dofs: fill mass/stiffness
        if( c.master() ) {
            res.master.push_back( c.dofs );

#if USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX
            if( !zero(c.H) ) add_shifted_right<Triplet,mat>( H_triplets, c.H, off_m );
            if( !zero(c.P) ) add_shifted_right<Triplet,mat>( P_triplets, c.P, off_m );
#else
			// scoped::timer step("independent dofs");
            mat shift = shift_right<mat>(off_m, c.size, _processed->size_m);
            if( !zero(c.H) ) res.H.middleRows(off_m, c.size) = res.H.middleRows(off_m, c.size) + c.H * shift;
            if( !zero(c.P) ) res.P.middleRows(off_m, c.size) = c.P * shift;
#endif

			// these should not be empty anyways
            if( !zero(c.b) ) res.b.segment(off_m, c.size) = c.b;
            if( !zero(c.v) ) res.v.segment(off_m, c.size) = c.v;

            off_m += c.size;
		}

		// mapped dofs
		else {

			if( !zero(Jc) ) {
                assert( Jc.cols() == int(_processed->size_m) );

                // actual response matrix mapping
                if( !zero(c.H) ) {

#if USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX
                    add_shifted_right( H_triplets, ltdl(Jc, c.H), 0 );
#else
                    res.H += ltdl(Jc, c.H);
#endif
                }

			}

			// compliant dofs: fill compliance/phi/lambda
            if( c.compliant() ) {
                res.compliant.push_back( c.dofs );
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

					SReal factor = 1.0 / (res.dt * l );


#if USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX
                    add_shifted_right( C_triplets, c.C, off_c, factor );
#else
                    res.C.middleRows(off_c, c.size) = c.C * shift_right<mat>(off_c, c.size, _processed->size_c, factor);
#endif
				}

				// phi
                res.phi.segment(off_c, c.size) = c.phi;

                if( c.lambda.size() ) res.lambda.segment( off_c, c.size ) = c.lambda;

                off_c += c.size;
			}
		}
	}

#if USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX
    res.H.setFromTriplets( H_triplets.begin(), H_triplets.end() );
    res.P.setFromTriplets( P_triplets.begin(), P_triplets.end() );
    res.C.setFromTriplets( C_triplets.begin(), C_triplets.end() );
#endif

    assert( off_m == _processed->size_m );
    assert( off_c == _processed->size_c );

	return res;
}


bool AssemblyVisitor::chunk::check() const {

	// let's be paranoid
	assert( dofs && size == dofs->getMatrixSize() );

	if( master() ) {
		assert( empty(C) );
		assert( !empty(P) );

//		assert( f.size() == int(size) );
		assert( v.size() == int(size) );

	} else {

		if(!empty(C)) {
			assert( phi.size() == int(size) );
			assert( damping >= 0 );
		}

	}

	// should be outer size ?
//	assert( empty(K) || K.rows() == int(size) );
//	assert( empty(M) || M.rows() == int(size) );

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
        std::for_each(prefix.rbegin(), prefix.rend(), propagation_helper(mparams, graph) );

		// TODO at this point it could be a good thing to prune
		// non-mechanical nodes in the graph, in order to avoid unneeded
		// mapping concatenations, then rebuild the prefix order

		start_node = 0;
	}
}







}
}

