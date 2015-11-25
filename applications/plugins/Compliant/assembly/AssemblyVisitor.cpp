#include "AssemblyVisitor.h"


#include <SofaEigen2Solver/EigenVectorWrapper.h>
#include <SofaBaseLinearSolver/SingleMatrixAccessor.h>
#include <SofaBaseLinearSolver/DefaultMultiMatrixAccessor.h>

#include <sofa/helper/cast.h>
#include "../utils/scoped.h"
#include "../utils/sparse.h"

#include "../constraint/ConstraintValue.h"
#include "../constraint/Stabilization.h"

using std::cerr;
using std::endl;


namespace sofa {
namespace simulation {

using namespace component::linearsolver;
using namespace core::behavior;


AssemblyVisitor::AssemblyVisitor(const core::MechanicalParams* mparams)
	: base( mparams ),
      mparams( mparams ),
	  start_node(0),
	  _processed(0)
{
    mparamsWithoutStiffness = *mparams;
    mparamsWithoutStiffness.setKFactor(0);
}


AssemblyVisitor::~AssemblyVisitor()
{
	if( _processed ) delete _processed;
}


AssemblyVisitor::chunk::chunk()
	: offset(0),
	  size(0),
      C(NULL),
      Ktilde(NULL),
	  mechanical(false),
	  vertex(-1),
      dofs(NULL) {

}

// pretty prints a mapping
static inline std::string mapping_name(simulation::Node* node) {
	return node->mechanicalMapping->getName() + " (class: " + node->mechanicalMapping->getClassName() + ") ";
}

// mapping informations as a map (parent dofs -> J matrix )
AssemblyVisitor::chunk::map_type AssemblyVisitor::mapping(simulation::Node* node) {
	chunk::map_type res;

	if( !node->mechanicalMapping ) return res;

	using helper::vector;

	assert( node->mechanicalMapping->getTo().size() == 1 &&
	        "only n -> 1 mappings are handled");

    ForceMaskActivate(node->mechanicalMapping->getMechTo());
    ForceMaskActivate(node->mechanicalMapping->getMechFrom());

    const vector<sofa::defaulttype::BaseMatrix*>* js = node->mechanicalMapping->getJs();

    ForceMaskDeactivate(node->mechanicalMapping->getMechTo());


    vector<core::BaseState*> from = node->mechanicalMapping->getFrom();

    for( unsigned i = 0, n = from.size(); i < n; ++i ) {

		// parent dofs
		dofs_type* p = safe_cast<dofs_type>(from[i]);

		// skip non-mechanical dofs
        if(!p || p->getSize()==0 ) continue;

        if( !notempty((*js)[i]) )
        {
            msg_warning("AssemblyVisitor" ) << "Empty mapping block for " << mapping_name(node) << " (is mapping matrix assembled?)";
            continue;
        }

		// mapping wrt p
        chunk::mapped& c = res[p];

        // getting BaseMatrix pointer
        c.J = (*js)[i];
	}

	return res;
}






// projection matrix
AssemblyVisitor::rmat AssemblyVisitor::proj(simulation::Node* node) {
	assert( node->mechanicalState );

	unsigned size = node->mechanicalState->getMatrixSize();

	// identity matrix TODO alloc ?
    tmp_p.compressedMatrix = shift_right<rmat>(0, size, size);

	for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++){
		node->projectiveConstraintSet[i]->projectMatrix(&tmp_p, 0);
        isPIdentity = false;
	}

    tmp_p.compressedMatrix.prune(0, 0);
	return tmp_p.compressedMatrix;
}




const defaulttype::BaseMatrix* compliance_impl( const core::MechanicalParams* mparams, BaseForceField* ffield )
{
    const defaulttype::BaseMatrix* c = ffield->getComplianceMatrix(mparams);

    if( notempty(c) )
    {
        return c;
    }
    else
    {
        msg_warning("AssemblyVisitor") << "compliance: "<<ffield->getName()<< "(node="<<ffield->getContext()->getName()<<"): getComplianceMatrix not implemented";
        // TODO inverting stiffness matrix
    }

    return NULL;
}


// compliance matrix
const defaulttype::BaseMatrix* AssemblyVisitor::compliance(simulation::Node* node)
{
    for(unsigned i = 0; i < node->forceField.size(); ++i )
    {
		BaseForceField* ffield = node->forceField[i];

		if( !ffield->isCompliance.getValue() ) continue;

        return compliance_impl( mparams, ffield );
	}

    for(unsigned i = 0; i < node->interactionForceField.size(); ++i )
    {
        BaseInteractionForceField* ffield = node->interactionForceField[i];

        if( !ffield->isCompliance.getValue() ) continue;

        if( ffield->getMechModel1() != ffield->getMechModel2() )
        {
            msg_warning("AssemblyVisitor") << "interactionForceField "<<ffield->getName()<<" cannot be simulated as a compliance.";
        }
        else
        {
            return compliance_impl( mparams, ffield );
        }
    }

    return NULL;
}


// geometric stiffness matrix
const defaulttype::BaseMatrix* AssemblyVisitor::geometricStiffness(simulation::Node* node)
{
//    std::cerr<<"AssemblyVisitor::geometricStiffness "<<node->getName()<<" "<<node->mechanicalMapping->getName()<<std::endl;

    core::BaseMapping* mapping = node->mechanicalMapping;
    if( mapping )
    {
        const sofa::defaulttype::BaseMatrix* k = mapping->getK();
        if( k ) return k;
    }

    return NULL;
}



// interaction forcefied in a node w/o a mechanical state
void AssemblyVisitor::interactionForceField(simulation::Node* node)
{
    for(unsigned i = 0; i < node->interactionForceField.size(); ++i )
    {
        BaseInteractionForceField* ffield = node->interactionForceField[i];

        if( ffield->getMechModel1() != ffield->getMechModel2() )
        {
            typedef EigenBaseSparseMatrix<SReal> BigSqmat;
            unsigned bigsize = ffield->getMechModel1()->getMatrixSize() + ffield->getMechModel2()->getMatrixSize();
            BigSqmat bigSqmat( bigsize, bigsize );
            DefaultMultiMatrixAccessor accessor;
            accessor.setGlobalMatrix( &bigSqmat );
            accessor.addMechanicalState( ffield->getMechModel1() );
            accessor.addMechanicalState( ffield->getMechModel2() );

            // an interactionFF is always a stiffness
            ffield->addMBKToMatrix( mparams, &accessor );
            bigSqmat.compress();

            if( !zero(bigSqmat.compressedMatrix) )
                interactionForceFieldList.push_back( InteractionForceField(bigSqmat.compressedMatrix,ffield) );


//            std::cerr<<"AssemblyVisitor::interactionForceField "<<ffield->getMechModel1()->getName()<<" "<<ffield->getMechModel2()->getName()<<" "<<bigSqmat<<std::endl;
        }
    }
}



// ode matrix
AssemblyVisitor::rmat AssemblyVisitor::odeMatrix(simulation::Node* node)
{
    unsigned size = node->mechanicalState->getMatrixSize();

    typedef EigenBaseSparseMatrix<SReal> Sqmat;
    Sqmat sqmat( size, size );

    for(unsigned i = 0; i < node->interactionForceField.size(); ++i )
    {
        BaseInteractionForceField* ffield = node->interactionForceField[i];

        if( ffield->getMechModel1() != ffield->getMechModel2() )
        {
//            std::cerr<<SOFA_CLASS_METHOD<<"WARNING: interactionForceField "<<ffield->getName()<<" will be treated as explicit, external forces (interactionForceFields are not handled by Compliant assembly, the same scene should be modelised with MultiMappings)"<<std::endl;

            typedef EigenBaseSparseMatrix<SReal> BigSqmat;
            unsigned bigsize = ffield->getMechModel1()->getMatrixSize() + ffield->getMechModel2()->getMatrixSize();
            BigSqmat bigSqmat( bigsize, bigsize );
            DefaultMultiMatrixAccessor accessor;
            accessor.setGlobalMatrix( &bigSqmat );
            accessor.addMechanicalState( ffield->getMechModel1() );
            accessor.addMechanicalState( ffield->getMechModel2() );

            // an interactionFF is always a stiffness
            ffield->addMBKToMatrix( mparams, &accessor );
            bigSqmat.compress();

            if( !zero(bigSqmat.compressedMatrix) )
                interactionForceFieldList.push_back( InteractionForceField(bigSqmat.compressedMatrix,ffield) );


//            std::cerr<<"AssemblyVisitor::odeMatrix "<<ffield->getName()<<" "<<bigSqmat<<std::endl;
        }
        else
        {
            // interactionForceFields that work on a unique set of dofs are OK
            SingleMatrixAccessor accessor( &sqmat );

            // when it is a compliant, you need to add M if mass, B but not K
            ffield->addMBKToMatrix( ffield->isCompliance.getValue() ? &mparamsWithoutStiffness : mparams, &accessor );
        }
    }

    // note that mass are included in forcefield
    for(unsigned i = 0; i < node->forceField.size(); ++i )
    {
        BaseForceField* ffield = node->forceField[i];

        SingleMatrixAccessor accessor( &sqmat );

        // when it is a compliant, you need to add M if mass, B but not K
        ffield->addMBKToMatrix( ffield->isCompliance.getValue() ? &mparamsWithoutStiffness : mparams, &accessor );
    }

    sqmat.compress();
    return sqmat.compressedMatrix;
}


void AssemblyVisitor::top_down(simulation::Visitor* vis) const {
	assert( !prefix.empty() );

	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
        simulation::Node* node = down_cast<simulation::Node>(graph[ prefix[i] ].data->dofs->getContext());
		vis->processNodeTopDown( node );
	}

}

void AssemblyVisitor::bottom_up(simulation::Visitor* vis) const {
	assert( !prefix.empty() );

	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {
        simulation::Node* node = down_cast<simulation::Node>(graph[ prefix[ n - 1 - i] ].data->dofs->getContext());
		vis->processNodeBottomUp( node );
	}
	
}





// this is called on the *top-down* traversal, once for each node. we
// simply fetch infos for each dof.
void AssemblyVisitor::fill_prefix(simulation::Node* node) {

    helper::ScopedAdvancedTimer advancedTimer( "assembly: fill_prefix" );

	assert( node->mechanicalState );
    assert( chunks.find( node->mechanicalState ) == chunks.end() && "Did you run the simulation with a DAG traversal?" );

    if( node->mechanicalState->getSize()==0 ) return;

	// fill chunk for current dof
	chunk& c = chunks[ node->mechanicalState ];

	c.size = node->mechanicalState->getMatrixSize();
	c.dofs = node->mechanicalState;

    vertex v; v.data = &c;

	c.H = odeMatrix( node );
//    cerr << "AssemblyVisitor::fill_prefix, c.H = " << endl << dmat(c.H) << endl;

     if( !zero(c.H) ) {
        c.mechanical = true;
     }

    // if the visitor is excecuted from a mapped node, do not look at its mapping
    if( node != start_node ) c.map = mapping( node );
	
	c.vertex = boost::add_vertex(graph);
	graph[c.vertex] = v;

	// independent dofs
	if( c.map.empty() ) {
		c.P = proj( node );
	} else {
		// mapped dofs

        // compliance
		c.C = compliance( node );
        if( notempty(c.C) ) {
			c.mechanical = true;
		}

        // geometric stiffness
        c.Ktilde = geometricStiffness( node );
	}
}



// bottom-up: build dependency graph
void AssemblyVisitor::fill_postfix(simulation::Node* node) {

    helper::ScopedAdvancedTimer advancedTimer( "assembly: fill_postfix" );

	assert( node->mechanicalState );

    if( node->mechanicalState->getSize()==0 ) return;

	assert( chunks.find( node->mechanicalState ) != chunks.end() );

	// fill chunk for current dof
	chunk& c = chunks[ node->mechanicalState ];

	for(chunk::map_type::const_iterator it = c.map.begin(), end = c.map.end();
	    it != end; ++it) {

        if( chunks.find(it->first) == chunks.end() ) continue; // this mechanical object is out of scope (ie not in the sub-graph controled by this solver)
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
         << "Ktilde:" << endl << Ktilde << endl
	     << "map: " << endl
		;

	for(map_type::const_iterator mi = map.begin(), me = map.end();
	    mi != me; ++mi) {
		cout << "from: " << mi->first->getName() << endl
		     << "J: " << endl
             << mi->second.J << endl
			;
	}
}

void AssemblyVisitor::debug() const {

	for(chunks_type::const_iterator i = chunks.begin(), e = chunks.end(); i != e; ++i ) {
		i->second.debug();
	}

}



// this is used to propagate mechanical flag upwards mappings (prefix
// in the graph order)
struct AssemblyVisitor::propagation_helper {

	const core::MechanicalParams* mparams;
	graph_type& g;

	propagation_helper(const core::MechanicalParams* mparams,
                       graph_type& g) : mparams(mparams), g(g) {}

    void operator()( unsigned v ) const {

		chunk* c = g[v].data;

        // if the current child is a mechanical dof
        // or if the current mapping is bringing geometric stiffness
        if( c->mechanical || notempty(c->Ktilde) ) {

            // have a look to all its parents
			for(graph_type::out_edge_range e = boost::out_edges(v, g);
                e.first != e.second; ++e.first) {

				chunk* p = g[ boost::target(*e.first, g) ].data;
                p->mechanical = true; // a parent of a mechanical child is necessarily mechanical
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
    scoped::timer step("assembly: mapping processing");

    process_type* res = new process_type();

    unsigned& size_m = res->size_m;
    unsigned& size_c = res->size_c;

	// independent dofs offsets (used for shifting parent)
    offset_type& offsets = res->offset.master;

	unsigned off_m = 0;
	unsigned off_c = 0;

	for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

		const chunk* c = graph[ prefix[i] ].data;

		// independent
		if( c->master() ) {
            offsets[ c->dofs ] = off_m;
			off_m += c->size;
        } else if( notempty(c->C) ) {
			off_c += c->size;
		}

	}

	// update total sizes
	size_m = off_m;
	size_c = off_c;

    // prefix mapping concatenation and stuff
    std::for_each(prefix.begin(), prefix.end(), process_helper(*res, graph) ); 	// TODO merge with offsets computation ?


    // special treatment for interaction forcefields
    fullmapping_type& full = res->fullmapping;
    for( InteractionForceFieldList::iterator it=interactionForceFieldList.begin(),itend=interactionForceFieldList.end();it!=itend;++it)
    {

        it->J.resize( it->H.rows(), size_m );

        rmat& Jp0 = full[ it->ff->getMechModel1() ];
        rmat& Jp1 = full[ it->ff->getMechModel2() ];

        if( empty(Jp0) ) {
            offset_type::const_iterator itoff = offsets.find(it->ff->getMechModel1());
            if( itoff != offsets.end() ) Jp0 = shift_right<rmat>( itoff->second, it->ff->getMechModel1()->getMatrixSize(), size_m);
        }
        if( empty(Jp1) ) {
            offset_type::const_iterator itoff = offsets.find(it->ff->getMechModel2());
            if( itoff != offsets.end() ) Jp1 = shift_right<rmat>( itoff->second, it->ff->getMechModel2()->getMatrixSize(), size_m);
        }

        if( !empty(Jp0) ) add( it->J, shift_left<rmat>( 0, it->ff->getMechModel1()->getMatrixSize(), it->H.rows() ) * Jp0 );
        if( !empty(Jp1) ) add( it->J, shift_left<rmat>( it->ff->getMechModel1()->getMatrixSize(), it->ff->getMechModel2()->getMatrixSize(), it->H.rows() ) * Jp1 );
    }

	return res;
}



// this is meant to optimize L^T D L products
inline const AssemblyVisitor::rmat& AssemblyVisitor::ltdl(const rmat& l, const rmat& d) const
{
    scoped::timer advancedTimer("assembly: ltdl");

//#ifdef _OPENMP
//    return component::linearsolver::mul_EigenSparseMatrix_MT( l.transpose(), component::linearsolver::mul_EigenSparseMatrix_MT( d, l ) );
//#else
    sparse::fast_prod(tmp1, d, l);
    tmp3 = l.transpose();
    sparse::fast_prod(tmp2, tmp3, tmp1);
    
    return tmp2;
//#endif
}


inline void AssemblyVisitor::add_ltdl(rmat& res, const rmat& l, const rmat& d)  const
{
    scoped::timer advancedTimer("assembly: ltdl");

    sparse::fast_prod(tmp1, d, l);
    tmp3 = l.transpose();
    sparse::fast_add_prod(res, tmp3, tmp1);
}



enum {
    METHOD_DEFAULT,
    METHOD_TRIPLETS,            // triplets seems fastest (for me) but
                                // might use too much memory
    METHOD_COEFREF,
    METHOD_DENSEMATRIX,
    METHOD_NOMULT
};

template<int Method = METHOD_NOMULT> struct add_shifted;

typedef AssembledSystem::rmat rmat;

template<> struct add_shifted<METHOD_TRIPLETS> {
    typedef Eigen::Triplet<SReal> Triplet;

    
    rmat& result;
    std::vector<Triplet> triplets;
    
    add_shifted(rmat& result) : result(result) {
        triplets.reserve(result.nonZeros() + result.rows());

        // don't forget to add prior values since we will overwrite
        // result in the dtor
        add_shifted_right<Triplet, rmat>( triplets, result, 0);
    }
    
    ~add_shifted() {
        result.setFromTriplets( triplets.begin(), triplets.end() );
    }

    template<class Matrix>
    void operator()(const Matrix& chunk, unsigned off, SReal factor = 1.0)  {
        add_shifted_right<Triplet, rmat>( triplets, chunk, off, factor);
    }

};





template<> struct add_shifted<METHOD_COEFREF> {

    rmat& result;
    add_shifted(rmat& result) : result(result) { }
    
    template<class Matrix>
    void operator()(const Matrix& chunk, unsigned off, SReal factor = 1.0) const {
        add_shifted_right<rmat>( result, chunk, off, factor );
    }

};


template<> struct add_shifted<METHOD_DENSEMATRIX> {

    typedef Eigen::Matrix<SReal, Eigen::Dynamic, Eigen::Dynamic> DenseMat;

    rmat& result;

    DenseMat dense;
    
    add_shifted(rmat& result)
        : result(result),
          dense( DenseMat::Zero(result.rows(), result.cols())) {

    }
    
    template<class Matrix>
    void operator()(const Matrix& chunk, unsigned off, SReal factor = 1.0)  {
        add_shifted_right<DenseMat,rmat>( dense, chunk, off, factor);
    }

    ~add_shifted() {
        convertDenseToSparse( result, dense );
    }

};

template<> struct add_shifted<METHOD_DEFAULT> {
    rmat& result;
    add_shifted(rmat& result)
        : result(result) {

    }

    // TODO optimize shift creation
    template<class Matrix>
    void operator()(const Matrix& chunk, unsigned off, SReal factor = 1.0) const {
        const rmat shift = shift_right<rmat>(off, chunk.cols(), result.cols(), factor);
        
        result.middleRows(off, chunk.rows()) = result.middleRows(off, chunk.rows()) + chunk * shift;
    }

};

// barely faster than DEFAULT
template<> struct add_shifted<METHOD_NOMULT> {
    rmat& result;
    add_shifted(rmat& result)
        : result(result) {

        // ideal would be somewhere between n and n^2 (TODO we should add a parameter)
        result.reserve( result.rows() );
    }

    template<class Matrix>
    void operator()(const Matrix& chunk, unsigned off, SReal factor = 1.0) const {
        result.middleRows(off, chunk.rows()) = result.middleRows(off, chunk.rows()) +
            shifted_matrix( chunk, off, result.cols(), factor);
    }

};



// produce actual system assembly
void AssemblyVisitor::assemble(system_type& res) const {
    scoped::timer step("assembly: build system");
	assert(!chunks.empty() && "need to send a visitor first");

	// assert( !_processed );

	// concatenate mappings and obtain sizes
    _processed = process();

	// result system
    res.reset(_processed->size_m, _processed->size_c);
    
	res.dt = mparams->dt();
    res.isPIdentity = isPIdentity;



    // Geometric Stiffness must be processed first, from mapped dofs to master dofs
    // warning, inverse order is important, to treat mapped dofs before master dofs
    // so mapped dofs can transfer their geometric stiffness to master dofs that will add it to the assembled matrix
    for( int i = (int)prefix.size()-1 ; i >=0 ; --i ) {

        const chunk& c = *graph[ prefix[i] ].data;
        assert( c.size );

        // only consider mechanical mapped dofs that have geometric stiffness
        if( !c.mechanical || c.master() || !c.Ktilde ) continue;

        // TODO remove copy for matrices that are already in the right type (EigenBaseSparseMatrix<SReal>)
        MySPtr<rmat> Ktilde( convertSPtr<rmat>( c.Ktilde ) );

        if( zero( *Ktilde ) ) continue;

        if( boost::out_degree(prefix[i],graph) == 1 ) // simple mapping
        {
//            std::cerr<<"simple mapping "<<c.dofs->getName()<<std::endl;
            // add the geometric stiffness to its only parent that will map it to the master level
            graph_type::out_edge_iterator parentIterator = boost::out_edges(prefix[i],graph).first;
            chunk* p = graph[ boost::target(*parentIterator, graph) ].data;
            add(p->H, mparams->kFactor() * *Ktilde ); // todo how to include rayleigh damping for geometric stiffness?

//            std::cerr<<"Assembly: "<<c.Ktilde<<std::endl;
        }
        else // multimapping
        {
            // directly add the geometric stiffness to the assembled level
            // by mapping with the specific jacobian from master to the (current-1) level


//            std::cerr<<"multimapping "<<c.dofs->getName()<<std::endl;

            // full mapping chunk for geometric stiffness
            const rmat& geometricStiffnessJc = _processed->fullmappinggeometricstiffness[ c.dofs ];



            //std::cerr<<geometricStiffnessJc<<std::endl;
//            std::cerr<<*Ktilde<<std::endl;
//            std::cerr<<Ktilde->nonZeros()<<std::endl;

//            std::cerr<<res.H.rows()<<" "<<geometricStiffnessJc.rows()<<std::endl;

            add_ltdl(res.H, geometricStiffnessJc, mparams->kFactor() * *Ktilde);
        }

    }


    // Then add interaction forcefields
    for( InteractionForceFieldList::iterator it=interactionForceFieldList.begin(),itend=interactionForceFieldList.end();it!=itend;++it)
    {
        add_ltdl(res.H, it->J, it->H);
    }




	// master/compliant offsets
	unsigned off_m = 0;
	unsigned off_c = 0;

    typedef add_shifted<> add_type;

    add_type add_H(res.H), add_P(res.P), add_C(res.C);

    const SReal c_factor = 1.0 /
        ( res.dt * res.dt * mparams->implicitVelocity() * mparams->implicitPosition() );
    
	// assemble system
    for( unsigned i = 0, n = prefix.size() ; i < n ; ++i ) {

		// current chunk
        const chunk& c = *graph[ prefix[i] ].data;
        assert( c.size );

        if( !c.mechanical ) continue;

		// independent dofs: fill mass/stiffness
        if( c.master() ) {
            res.master.push_back( c.dofs );

            if( !zero(c.H) ) add_H(c.H, off_m);
            if( !zero(c.P) ) add_P(c.P, off_m);
            
            off_m += c.size;
		}

		// mapped dofs
		else {

            // full mapping chunk
            const rmat& Jc = _processed->fullmapping[ c.dofs ];

			if( !zero(Jc) ) {
                assert( Jc.cols() == int(_processed->size_m) );

                // actual response matrix mapping
                if( !zero(c.H) ) add_H(ltdl(Jc, c.H), 0);
            }


			// compliant dofs: fill compliance/phi/lambda
			if( c.compliant() ) {
				res.compliant.push_back( c.dofs );
				// scoped::timer step("compliant dofs");
				assert( !zero(Jc) );


                // TODO remove copy for matrices that are already in the right type (EigenBaseSparseMatrix<SReal>)
                MySPtr<rmat> C( convertSPtr<rmat>( c.C ) );


                // fetch projector and constraint value if any
                AssembledSystem::constraint_type constraint;
                constraint.projector = c.dofs->getContext()->get<component::linearsolver::Constraint>( core::objectmodel::BaseContext::Local );
                constraint.value = c.dofs->getContext()->get<component::odesolver::BaseConstraintValue>( core::objectmodel::BaseContext::Local );

                // by default the manually given ConstraintValue is used
                // otherwise a fallback is used depending on the constraint type
                if( !constraint.value ) {

                    // a non-compliant (hard) bilateral constraint is stabilizable
                    if( zero(*C) /*|| fillWithZeros(*C)*/ ) constraint.value = new component::odesolver::Stabilization( c.dofs );
                    // by default, a compliant (elastic) constraint is not stabilized
                    else constraint.value = new component::odesolver::ConstraintValue( c.dofs );

                    c.dofs->getContext()->addObject( constraint.value );
                    constraint.value->init();
                }
                res.constraints.push_back( constraint );


				// mapping
				res.J.middleRows(off_c, c.size) = Jc;

                // compliance

                if( !zero( *C ) ) add_C(*C, off_c, c_factor);
                
				off_c += c.size;
			}
		}
	}

    assert( off_m == _processed->size_m );
    assert( off_c == _processed->size_c );

}

// TODO redo
bool AssemblyVisitor::chunk::check() const {

	// let's be paranoid
	assert( dofs && size == dofs->getMatrixSize() );

	if( master() ) {
        assert( !notempty(C) );
		assert( !empty(P) );

		// TODO size checks on M, J, ...

	} else {

        if(notempty(C)) {
            assert( C->rows() == int(size) );
		}

	}

	return true;
}


void AssemblyVisitor::clear() {

	chunks.clear();
	prefix.clear();
	graph.clear();

}



Visitor::Result AssemblyVisitor::processNodeTopDown(simulation::Node* node) {
    if( !start_node )
    {
        start_node = node;
        isPIdentity = true;
    }

	if( node->mechanicalState ) fill_prefix( node );
    else if( !node->interactionForceField.empty() ) interactionForceField( node );
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

