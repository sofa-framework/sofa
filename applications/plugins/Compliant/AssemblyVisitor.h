#ifndef ASSEMBLYVISITOR_H
#define ASSEMBLYVISITOR_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <map>

#include "AssembledSystem.h"
#include "utils/graph.h"

#include "AssemblyHelper.h"

#include "./utils/find.h"

namespace sofa {
namespace simulation {

// a visitor for system assembly: sending the visitor will fetch
// data, and actual system assembly is performed using
// ::assemble(), yielding an AssembledSystem

// TODO IMPORTANT: handle damping (including rayleigh) for material
// simulated as compliant constraints
		
// TODO preallocate global vectors for all members to avoid multiple,
// small allocs during visit (or tweak allocator ?)

// TODO a few map accesses may also be optimized here, e.g. using
// preallocated std::unordered_map instead of std::map for
// chunks/global, in case the scene really has a large number of
// mstates

// TODO shift matrices may also be improved using eigen magic
// instead of actual sparse matrices (causing allocs)
// USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX try another implementation
// building assembled matrces from sequentialy generated triplets
// but it is not proven that is more efficient



class AssemblyVisitor : public simulation::MechanicalVisitor {
protected:
	typedef simulation::MechanicalVisitor base;
    const core::MechanicalParams* mparams;
public:

	typedef SReal real;

	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
	typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;

	// default: row-major
	typedef rmat mat;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
			
    AssemblyVisitor(const core::MechanicalParams* mparams = 0, MultiVecDerivId velId = MultiVecDerivId(core::VecDerivId::velocity()), MultiVecDerivId lagrange = MultiVecDerivId() );
    virtual ~AssemblyVisitor();

//protected:
    MultiVecDerivId _velId;

public:
    MultiVecDerivId lagrange;
    simulation::Node* start_node;

	// collect data chunks during visitor execution
	virtual Visitor::Result processNodeTopDown(simulation::Node* node);
	virtual void processNodeBottomUp(simulation::Node* node);
	
	// reset state
	void clear();
	
	// distribute data over master dofs, in given vecid
    void distribute_master(core::behavior::MultiVecDeriv::MyMultiVecId id, const vec& data);

    // distribute data over compliant dofs, in given vecid
    void distribute_compliant(core::behavior::MultiVecDeriv::MyMultiVecId id, const vec& data);
			
	// outputs data to std::cout
	void debug() const; 
	
public:
	
	typedef core::behavior::BaseMechanicalState dofs_type;
	
	// data chunk for each dof
	struct chunk {
		chunk();
				
		unsigned offset, size;

        mat C; ///< Compliance matrix
        mat P; ///< Projective constraint matrix
        mat H; ///< linear combinaison of M,B,K (mass, damping, stiffness matrices)
				
		struct mapped {
            mat J; ///< mapping jacobian
            mat K; ///< geometric stiffness
		};

        vec b; ///< ode system right-hand term (including force sum)
        vec phi; ///< constraint values
        vec v; ///< actual velocity
        vec lambda; ///< Lagrange multipliers (previously computed)


        real damping; ///< global constraint damping

        bool tagged;

		// this is to remove f*cking mouse dofs
        bool mechanical; ///< is it a mechanical dof i.e. influenced by a mass or stiffness or compliance
		
		bool master() const { return mechanical && map.empty(); }
		bool compliant() const { return mechanical && phi.size(); }
		
		unsigned vertex;
		
		dofs_type* dofs;

        typedef std::map< dofs_type*, mapped> map_type;
        map_type map;
		
		// check consistency
		bool check() const;

		void debug() const;
	};

	vec vector(dofs_type* dofs, core::ConstVecId id); // get

	void vector(dofs_type*, core::VecId id, const vec::ConstSegmentReturnType& data); // set
			
public:

    mat compliance(simulation::Node* node);
	mat proj(simulation::Node* node);
    mat odeMatrix(simulation::Node* node);
			
	chunk::map_type mapping(simulation::Node* node);
			
	vec force(simulation::Node* node);
	
    vec vel(simulation::Node* node, MultiVecDerivId velId );
	vec phi(simulation::Node* node);
	vec lambda(simulation::Node* node);

	real damping(simulation::Node* node);

	// fill data chunk for node
    virtual void fill_prefix(simulation::Node* node);

    void fill_postfix(simulation::Node* node);

protected:

	// TODO hide this ! but adaptive stuff needs it
public:
	
	// TODO remove maps, stick everything in chunk ? again, adaptive
	// stuff might need it

	// full mapping/stiffness matrices
	typedef std::map<dofs_type*, mat> full_type;

	// dof offset
	typedef std::map<dofs_type*, unsigned> offset_type;
			
	struct process_type {
		unsigned size_m;
		unsigned size_c;
				
		full_type full;
				
		// offsets
		struct {
			offset_type master;
			offset_type compliant; // TODO
		} offset;
				
	};

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
	
	// TODO we don't even need dofs since they are in data
	struct vertex {
		dofs_type* dofs;
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
    system_type assemble() const;
	
private:

	// temporaries
	mutable vec tmp;

	// work around the nonexistent copy ctor of base class
	mutable struct tmp_p_type : component::linearsolver::EigenBaseSparseMatrix<real> {
		typedef component::linearsolver::EigenBaseSparseMatrix<real> base;
		tmp_p_type() : base() {} 
		tmp_p_type(const tmp_p_type&) : base() { }
	} tmp_p;

    //simulation::Node* start_node;

public:
    // do not perform entire assembly, but only compute momentum sys.p and constraint value sys.phi. The process_type p must have been computed from a previous call to assemble
    template<class SystemType>
    void updateRightHand( SystemType& sys ) {
        assert(!chunks.empty() && "need to send a visitor first");

        assert( _processed );

        // master/compliant offsets
        unsigned off_m = 0;
        unsigned off_c = 0;

        // assemble system
        for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

            // current chunk
            chunk& c = *graph[ prefix[i] ].data;
            assert( c.size );

            if( !c.mechanical ) continue;

            dofs_type* dofs = graph[ prefix[i] ].dofs;

            // to access new multivec data
            Node* node = static_cast<Node*>(dofs->getContext());

            if( c.master() )
            {
                sys.b.segment(off_m, c.size) = force( node );

                off_m += c.size;
            }
            else if(  c.compliant() )  //compliant dofs: fill compliance/phi/lambda
            {
                sys.phi.segment(off_c, c.size) = phi( node );

                off_c += c.size;
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

                    Jp = shift_right<mat>( find(offsets, vp.dofs), p->size, size_m);

                    // TODO optimize!
                    // filter absolute mapping to get filtered mapped mass/stiffness.
                    if(! zero(p->P) ) Jp = p->P * Jp;
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

}
}


#endif
