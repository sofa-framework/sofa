#ifndef ASSEMBLYVISITOR_H
#define ASSEMBLYVISITOR_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <map>

#include "AssembledSystem.h"
#include "utils/graph.h"

#include "AssemblyHelper.h"

namespace sofa {
namespace simulation {

// a visitor for system assembly: sending the visitor will fetch
// data, and actual system assembly is performed using
// ::assemble(), yielding an AssembledSystem
		
// TODO preallocate global vectors for all members to avoid multiple,
// small allocs during visit (or tweak allocator ?)

// TODO callgrind reports that some performance gains can be
// obtained during the fetching of mass/projection matrices (but
// mass, mainly), as the eigen/sofa matrix wrapper uses lots of
// maps insertions internally. 

// TODO a few map accesses may also be optimized here, e.g. using
// preallocated std::unordered_map instead of std::map for
// chunks/global, in case the scene really has a large number of
// mstates

// TODO shift matrices may also be improved using eigen magic
// instead of actual sparse matrices (causing allocs)
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
		mat M, K, C, P;
				
		struct mapped {
			mat J;
			mat K;
		};
				
		typedef std::map< dofs_type*, mapped> map_type;
		map_type map;
				
		// TODO only expose sofa data through eigen maps ? but ... casts ?

		vec f, v, phi, lambda;
		real damping;

        bool tagged;

		// this is to remove f*cking mouse dofs
		bool mechanical;
		
		bool master() const { return mechanical && map.empty(); }
		bool compliant() const { return mechanical && phi.size(); }
		
		unsigned vertex;
		
		dofs_type* dofs;
		
		// check consistency
		bool check() const;

		void debug() const;
	};

	vec vector(dofs_type* dofs, core::ConstVecId id); // get

	void vector(dofs_type*, core::VecId id, const vec::ConstSegmentReturnType& data); // set
			
public:
	mat mass(simulation::Node* node);

	
	mat compliance(simulation::Node* node);
	mat stiff(simulation::Node* node);
	mat proj(simulation::Node* node);
			
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
    void updateConstraintAndMomentum( MultiVecDerivId velId, SystemType& sys ) {
        assert(!chunks.empty() && "need to send a visitor first");

        assert( _processed );

        // master/compliant offsets
        unsigned off_m = 0;
        unsigned off_c = 0;

        sys.p.setZero();

        // assemble system
        for(unsigned i = 0, n = prefix.size(); i < n; ++i) {

            // current chunk
            chunk& c = *graph[ prefix[i] ].data;
            assert( c.size );

            if( !c.mechanical ) continue;

            dofs_type* dofs = graph[ prefix[i] ].dofs;
            const mat& Jc = _processed->full[ dofs ];

            // to access new multivec data
            Node* node = static_cast<Node*>(dofs->getContext());

            // independent dofs: fill mass/stiffness
            if( c.master() )
            {
                // mass matrix / momentum
                if( !zero(c.M) ) {
                    vec::SegmentReturnType r = sys.p.segment(off_m, c.size);

                    r.noalias() = r + c.M * vel( node, velId );
                }

                off_m += c.size;
            }
            else
            {
                if( !zero(Jc) ) {
                    assert( Jc.cols() == int(_processed->size_m) );

                    {
                        if( !zero(c.M) ) {
                            // contribute mapped mass
                            assert( c.v.size() == int(c.size) );
                            assert( c.M.cols() == int(c.size) );
                            assert( c.M.rows() == int(c.size) );

                            // momentum TODO avoid alloc
                            sys.p.noalias() = sys.p + Jc.transpose() * (c.M * vel( node, velId ));
                        }
                    }

                }

                // compliant dofs: fill compliance/phi/lambda
                if( c.compliant() ) {

                    // phi
                    sys.phi.segment(off_c, c.size) = phi( node );

                    off_c += c.size;
                }
            }
        }
    }

};
		
}
}


#endif
