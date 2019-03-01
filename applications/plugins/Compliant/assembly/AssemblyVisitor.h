#ifndef ASSEMBLYVISITOR_H
#define ASSEMBLYVISITOR_H

#include <Compliant/config.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <map>

#include "../utils/graph.h"
#include "../utils/find.h"

#include "AssembledSystem.h"
#include "AssemblyHelper.h"

namespace sofa {
namespace simulation {

// a visitor for system assembly: sending the visitor will fetch
// data, and actual system assembly is performed using
// ::assemble(), yielding an AssembledSystem
		
// TODO preallocate global vectors for all members to avoid multiple,
// small allocs during visit (or tweak allocator ?)

// TODO a few map accesses may also be optimized here, e.g. using
// preallocated std::unordered_map instead of std::map for
// chunks/global, in case the scene really has a large number of
// mstates

class SOFA_Compliant_API AssemblyVisitor : public simulation::MechanicalVisitor {
protected:
    typedef simulation::MechanicalVisitor base;
public:

	// would it be better to dynamically modified mparams ?
    const core::MechanicalParams* mparams;
    core::MechanicalParams mparamsWithoutStiffness;

	typedef SReal real;

	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
    typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dmat;

    // default: row-major
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
			
    AssemblyVisitor(const core::MechanicalParams* mparams);
    ~AssemblyVisitor() override;

//protected:
//    MultiVecDerivId _velId;

	// applies another visitor topDown/bottomUp based on internal
	// graph. this is to work around bogus graph traversal in case of
	// multi mappings

	void top_down(simulation::Visitor* vis) const;
	void bottom_up(simulation::Visitor* vis) const;

public:
	simulation::Node* start_node;

	// collect data chunks during visitor execution
	Visitor::Result processNodeTopDown(simulation::Node* node) override;
	void processNodeBottomUp(simulation::Node* node) override;
	
	// reset state
	void clear();
	
	// outputs data to std::cout
	void debug() const; 
	
public:
	
	typedef core::behavior::BaseMechanicalState dofs_type;
	
	// data chunk for each dof
	struct chunk {
		chunk();
				
        unsigned offset;
        unsigned size; // dof type size

        const defaulttype::BaseMatrix* C; ///< Compliance matrix (only valid for mapped dof with a compliance)
        rmat P; ///< Projective constraint matrix (only valid for master dof)
        rmat H; ///< linear combinaison of M,B,K (mass, damping, stiffness matrices)
        const defaulttype::BaseMatrix* Ktilde; ///< geometric stiffness (only valid for mapped dof) @warning: size=parent*parent
				
		struct mapped {
            mapped() : J(NULL) {}
            const defaulttype::BaseMatrix* J; ///< mapping jacobian
		};

        /// flagged as a mechanical dof
        /// i.e. influenced by a mass or damping or stiffness or compliance
        /// Note it includes having a mechanical child
        /// or a child mapped with a mapping generating geometric stiffness
        bool mechanical;
		
		bool master() const { return mechanical && map.empty(); }
        bool compliant() const { return mechanical && notempty(C); }
		
		unsigned vertex;
		
		// TODO SPtr here ?
		dofs_type* dofs;

		typedef std::map< dofs_type*, mapped> map_type;
		map_type map;
		
		// check consistency
		bool check() const;

		void debug() const;
	};

    // a special structure to handle InteractionForceFields
    struct InteractionForceField
    {
        InteractionForceField( rmat H, core::behavior::BaseInteractionForceField* ff ) : H(H), ff(ff) {
//        std::cerr<<"Assembly InteractionForceField "<<H<<std::endl;
        }
        rmat H; ///< linear combinaison of M,B,K (mass, damping, stiffness matrices)
        core::behavior::BaseInteractionForceField* ff;
        rmat J;
    };
    typedef std::list<InteractionForceField> InteractionForceFieldList;
    mutable InteractionForceFieldList interactionForceFieldList;

public:

    const defaulttype::BaseMatrix* compliance(simulation::Node* node);
    const defaulttype::BaseMatrix* geometricStiffness(simulation::Node* node);
    rmat proj(simulation::Node* node);
    rmat odeMatrix(simulation::Node* node);
    void interactionForceField(simulation::Node* node);
			
	chunk::map_type mapping(simulation::Node* node);
			
	// fill data chunk for node
	virtual void fill_prefix(simulation::Node* node);
	
	void fill_postfix(simulation::Node* node);

protected:

	// TODO hide this ! but adaptive stuff needs it
public:
	
	// TODO remove maps, stick everything in chunk ? again, adaptive
	// stuff might need it

    /// full mapping matrices
    /// for a dof, gives it full mapping from master level
    typedef std::map<dofs_type*, rmat> fullmapping_type;

	// dof offset
	typedef std::map<dofs_type*, unsigned> offset_type;
			
	struct process_type {
		unsigned size_m;
		unsigned size_c;
				
        fullmapping_type fullmapping; ///< full mapping from a dof to the master level
        fullmapping_type fullmappinggeometricstiffness; ///< full mapping from a dof-1 level to the master level (used to map the geometric stiffness)
				
		// offsets
		struct {
			offset_type master;
			offset_type compliant; // TODO
		} offset;
				
	};

	// max: wtf is this ?
	mutable process_type *_processed;

	// builds global mapping / full stiffness matrices + sizes
    virtual process_type* process() const;
			
	// helper functors
	struct process_helper;
	struct propagation_helper;
	struct prefix_helper;

	// data chunks
	typedef std::map< dofs_type*, chunk > chunks_type;
	mutable chunks_type chunks;

	// traversal order
	typedef std::vector< unsigned > prefix_type;
	prefix_type prefix;

    struct vertex {
		chunk* data;					// avoids map lookups 
	};

	struct edge {
		const chunk::mapped* data;
	};
	
	typedef utils::graph<vertex, edge, boost::bidirectionalS> graph_type;
	graph_type graph;

public:


	// build assembled system (needs to send visitor first)
	// if the pp pointer is given, the created process_type structure will be kept (won't be deleted)
	typedef component::linearsolver::AssembledSystem system_type;
	void assemble(system_type& ) const;

    
private:

	// temporaries
	mutable vec tmp;

	// work around the nonexistent copy ctor of base class
	mutable struct tmp_p_type : component::linearsolver::EigenBaseSparseMatrix<real> {
		typedef component::linearsolver::EigenBaseSparseMatrix<real> base;
		tmp_p_type() : base() {} 
		tmp_p_type(const tmp_p_type&) : base() { }
	} tmp_p;

    bool isPIdentity; ///< false iff there are projective constraints

    //simulation::Node* start_node;


    // keep temporaries allocated
    mutable rmat tmp1, tmp2, tmp3;


    // this is meant to optimize L^T D L products
    const rmat& ltdl(const rmat& l, const rmat& d) const;
    void add_ltdl(rmat& res, const rmat& l, const rmat& d) const;

};





/// Computing the full jacobian matrices from masters to every mapped dofs
/// ie multiplies mapping matrices together for everyone in the graph
// TODO why is this here?
// -> because we need an access to it when deriving AssemblyVisitor
// -> could be moved in AssemblyHelper?
// -> or at least its implementation could be written in the .cpp
struct AssemblyVisitor::process_helper {

    process_type& res;
    const graph_type& g;

    process_helper(process_type& res, const graph_type& g)
        : res(res), g(g)  {

    }

    void operator()(unsigned v) const {

        chunk* c = g[v].data;
        dofs_type* curr = c->dofs;

        const unsigned& size_m = res.size_m;
        fullmapping_type& full = res.fullmapping;
        offset_type& offsets = res.offset.master;

        if( c->master() || !c->mechanical ) return;

        rmat& Jc = full[ curr ]; // (output) full mapping from independent dofs to mapped dofs c
        assert( empty(Jc) );


        // full jacobian for multimapping's geometric stiffness
        rmat* geometricStiffnessJc = NULL;
        unsigned localOffsetParentInMapped = 0; // only used for multimappings
        if( boost::out_degree(v,g)>1 && notempty(c->Ktilde) )
        {
            geometricStiffnessJc = &res.fullmappinggeometricstiffness[ curr ];
        }


        for( graph_type::out_edge_range e = boost::out_edges(v, g); e.first != e.second; ++e.first) {

            vertex vp = g[ boost::target(*e.first, g) ];

            // parent data chunk/mapping matrix
            const chunk* p = vp.data;
            dofs_type* pdofs = p->dofs;

            rmat& Jp = full[ pdofs ]; // (input) full mapping from independent dofs to parent p of dofs c
            {
                // mapping blocks
                helper::OwnershipSPtr<rmat> jc( convertSPtr<rmat>( g[*e.first].data->J ) );

                if( zero( *jc ) )
                {
                    continue;

                    // Note a Jacobian can be null in a multimapping (child only mapped from one parent)
                    // or when the corresponding mask is empty
//                    MAINLOGGER( Info, "Empty Jacobian for mapping: "<<((simulation::Node*)curr->getContext())->mechanicalMapping->getPathName()
//                                              << std::endl << "mask=\""<<curr->forceMask <<"\"", "AssemblyVisitor" )
                }


                // parent is not mapped: we put a shift matrix with the
                // correct offset as its full mapping matrix, so that its
                // children will get the right place on multiplication
                if( p->master() && empty(Jp) ) {
                    Jp = shift_right<rmat>( find(offsets, pdofs), p->size, size_m);
                }

                // Jp can be empty for multinodes, when a child is mapped only from a subset of its parents
                if(!empty(Jp) ){
                    // scoped::timer step("mapping matrix product");

                    // TODO optimize this, it is the most costly part
                    add_prod(Jc, *jc, Jp ); // full mapping

                    if( geometricStiffnessJc )
                    {
                        // mapping for geometric stiffness
                        add_prod( *geometricStiffnessJc,
                                  shift_left<rmat>( localOffsetParentInMapped,
                                                   p->size,
                                                   c->Ktilde->rows() ),
                                  Jp );
                        localOffsetParentInMapped += p->size;
                    }
                }
            }

//            std::cerr<<"Assembly::geometricStiffnessJc "<<geometricStiffnessJc<<" "<<curr->getName()<<std::endl;
        }

//        if( zero(Jc) && curr->getSize() !=0 )  {
//            // If the dof size is null, let's consider it is not a big deal.
//            // Indeed, having null dof is useful when setting up a static graph that is filled in dynamically

//            using namespace std;
//            MAINLOGGER( Warning,
//                        "Houston we have a problem. Null full Jacobian for dof \""<< c->dofs->getPathName()<<"\""<< endl
//                        << "master=" << c->master() << ", "
//       //                 << "mapped: " << (c->map.empty() ? string("nope") : p->dofs->getName() )<< endl
//       //                 << "p mechanical ? " << p->mechanical << endl
//       //                 << "empty Jp " << empty(Jp) << endl
//                        << "empty Jc=" << empty(Jc) << ", "
//                        , "AssemblyVisitor" )

//            assert( false );
//        }
//        else
//            MAINLOGGER( Info, ((simulation::Node*)c->dofs->getContext())->mechanicalMapping->getPathName()<<" "<<Jc.nonZeros() , "AssemblyVisitor" )

    }


};

}
}


#endif
