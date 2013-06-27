#ifndef ASSEMBLYVISITOR_H
#define ASSEMBLYVISITOR_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <map>

#include "AssembledSystem.h"
#include "utils/graph.h"

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
	typedef simulation::MechanicalVisitor base;
			
	const core::MechanicalParams* mparams;
public:

	typedef SReal real;

	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
	typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;

	// default: row-major
	typedef rmat mat;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
			
	AssemblyVisitor(const core::MechanicalParams* mparams = 0);
			
	// collect data chunks during visitor execution
	virtual Visitor::Result processNodeTopDown(simulation::Node* node);
	virtual void processNodeBottomUp(simulation::Node* node);
	
	// reset state
	void clear();

	// WARNING: EXPLICIT/IMPLICIT KEYWORDS ARE BUILT-IN MACROS ON WINDOWS
	typedef enum {
		OPTION_EXPLICIT,
		OPTION_IMPLICIT
	} assemble_option;
	
	// builds assembled system (needs to send visitor first)
	typedef component::linearsolver::AssembledSystem system_type;
	system_type assemble() const;
			

	// distribute data over master dofs
	void distribute_master(core::VecId id, const vec& data);

	// distribute data over compliant dofs
	void distribute_compliant(core::VecId id, const vec& data);
	void distribute_compliant(core::behavior::MultiVecDeriv::MyMultiVecId id, const vec& data);
			
	// outputs data to std::cout
	void debug() const; 

	// TODO wrap this
	typedef core::behavior::MultiVecDeriv::MyMultiVecId lagrange_type;
	lagrange_type lagrange;
	
public:
	
	typedef core::behavior::BaseMechanicalState dofs_type;
			
	// data chunk for each dofs
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
				
		// TODO only expose sofa data through eigen maps ?

		vec f, v, phi, lambda;

		system_type::flags_type flags;

		// extra info, hacks go here :)
		typedef system_type::block::data_type extra_type;
		extra_type extra;
		
		real damping;

		// this is to remove f*cking mouse dofs
		bool mechanical;
		
		bool master() const { return mechanical && map.empty(); }
		bool compliant() const { return mechanical && phi.size(); }
		
		unsigned vertex;
		
		dofs_type* dofs;
		
		// check consistency
		bool check() const;
	};

	static mat convert( const defaulttype::BaseMatrix* m);
	vec vector(simulation::Node* node, core::ConstVecId id); // get

	void vector(dofs_type*, core::VecId id, const vec::ConstSegmentReturnType& data); // set
			
public:
	mat mass(simulation::Node* node);

	
	mat compliance(simulation::Node* node);
	mat stiff(simulation::Node* node);
	mat proj(simulation::Node* node);
			
	chunk::map_type mapping(simulation::Node* node);
			
	vec force(simulation::Node* node);
	
	chunk::extra_type extra(simulation::Node* node);

	system_type::flags_type flags(simulation::Node* node);
	
	vec vel(simulation::Node* node);
	vec phi(simulation::Node* node);
	vec lambda(simulation::Node* node);

	// TODO is this needed ?
	real damping(simulation::Node* node);

	// fill data chunk for node
	void fill_prefix(simulation::Node* node);
	void fill_postfix(simulation::Node* node);

protected:

	// TODO hide
public:
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


	// builds global mapping / full stiffness matrices + sizes
	// TODO pass prefix/chunks as parameters
	virtual process_type process() const;
			
	struct process_helper;
	struct propagation_helper;
	struct prefix_helper;

	// data chunks
	typedef std::map< dofs_type*, chunk > chunks_type;
	mutable chunks_type chunks;

	static void debug_chunk(const chunks_type::const_iterator& );

	// traversal order
	typedef std::vector< dofs_type* > prefix_type;
	mutable prefix_type prefix;
			
	// mapping graph
	struct vertex {
		dofs_type* dofs;
	};

	struct edge {
		const chunk::mapped* data;
	};
	
	typedef utils::graph<vertex, edge, boost::bidirectionalS> graph_type;
	graph_type graph;
	
private:
	// total dofs size
	// unsigned size;
			
	// temporaries
	mutable vec tmp;

	// work around the nonexistent copy ctor of base class
	mutable struct tmp_p_type : component::linearsolver::EigenBaseSparseMatrix<real> {
		typedef component::linearsolver::EigenBaseSparseMatrix<real> base;
		tmp_p_type() : base() {} 
		tmp_p_type(const tmp_p_type&) : base() { }
	} tmp_p;

	simulation::Node* start_node;

};
		
}
}


#endif
