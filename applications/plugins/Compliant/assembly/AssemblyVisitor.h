#ifndef ASSEMBLYVISITOR_H
#define ASSEMBLYVISITOR_H

#include "initCompliant.h"
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaEigen2Solver/EigenVector.h>
#include <map>

#include "AssembledSystem.h"
#include "../utils/graph.h"

#include "AssemblyHelper.h"

#include "../utils/find.h"


// select the way to perform shifting of local matrix in a larger matrix, default = build a shift matrix and be multiplied with
#define USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX 0 // more memory and not better
#define USE_SPARSECOEFREF_RATHER_THAN_SHIFT_MATRIX 0 // bof
#define USE_DENSEMATRIX_RATHER_THAN_SHIFT_MATRIX 0 // very slow
#define SHIFTING_MATRIX_WITHOUT_MULTIPLICATION 1 // seems a bit faster



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

// TODO shift matrices may also be improved using eigen magic
// instead of actual sparse matrices (causing allocs)
// USE_TRIPLETS_RATHER_THAN_SHIFT_MATRIX try another implementation
// building assembled matrces from sequentialy generated triplets
// but it is not proven that is more efficient


/// compute forces only for compliant forcefields (after reseting the mapped dof forces and accumulate them toward the independant dofs)
class MechanicalComputeComplianceForceVisitor : public MechanicalComputeForceVisitor
{
public:
    MechanicalComputeComplianceForceVisitor(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res )
        : MechanicalComputeForceVisitor(mparams,res,true)
    {
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/) { return RESULT_CONTINUE; }
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->params /* PARAMS FIRST */, res.getId(mm));
        return RESULT_CONTINUE;
    }
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if( ff->isCompliance.getValue() ) ff->addForce(this->mparams, res);
        return RESULT_CONTINUE;
    }

};


/// res += constraint forces (== lambda/dt), only for mechanical object linked to a compliance
class MechanicalAddComplianceForce : public MechanicalVisitor
{
    core::MultiVecDerivId res, lambdas;
    SReal invdt;


public:
    MechanicalAddComplianceForce(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res, core::MultiVecDerivId lambdas, SReal dt )
        : MechanicalVisitor(mparams), res(res), lambdas(lambdas), invdt(1.0/dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    // reset lambda where there is no compliant FF
    // these reseted lambdas were previously propagated, but were not computed from the last solve
    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        // a compliant FF must be alone, so if there is one, it is the first one of the list.
        const core::behavior::BaseForceField* ff = NULL;

        if( !node->forceField.empty() ) ff = *node->forceField.begin();
        else if( !node->interactionForceField.empty() ) ff = *node->interactionForceField.begin();

        if( !ff || !ff->isCompliance.getValue() )
        {
            core::behavior::BaseMechanicalState* mm = ff->getContext()->getMechanicalState();
            const core::VecDerivId& lambdasid = lambdas.getId(mm);
            if( !lambdasid.isNull() ) // previously allocated
            {
                mm->resetForce(this->params, lambdasid);
            }
        }
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        return fwdMechanicalState( node, mm );
    }

    // pop-up lamdas without modifying f
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        map->applyJT( this->mparams, lambdas, lambdas );
    }

    // for all dofs, f += lamda / dt
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        const core::VecDerivId& lambdasid = lambdas.getId(mm);
        if( !lambdasid.isNull() ) // previously allocated
        {
            const core::VecDerivId& resid = res.getId(mm);

//            mm->vOp( this->params, resid, resid, lambdasid, invdt ); // f += lambda / dt

            // hack to improve stability and energy preservation
            // TO BE STUDIED: lambda must be negative to generate geometric stiffness
            // TODO find a more efficient way to implemented it
            const size_t dim = mm->getMatrixSize();
            component::linearsolver::AssembledSystem::vec buffer( dim );
            mm->copyToBuffer( &buffer(0), lambdasid, dim );
            for( size_t i=0 ; i<dim ; ++i )
                if( buffer[i] > 0 ) buffer[i] *= -invdt;
                else buffer[i] *= invdt; // constraint_force = lambda / dt
            mm->addFromBuffer( resid, &buffer(0), dim ); // f += lambda / dt
        }
    }

    virtual void bwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        bwdMechanicalState( node, mm );
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalAddLambdas";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+res.getName()+","+lambdas.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
        addWriteVector(lambdas);
    }
#endif
};


/// compute compliant forces fc and stiffness forces fk (after reseting the mapped dof forces and accumulate them toward the independant dofs)
class MechanicalComputeStiffnessAndComplianceForcesVisitor : public MechanicalVisitor
{

    core::MultiVecDerivId fk,fc;

public:

    MechanicalComputeStiffnessAndComplianceForcesVisitor(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId fk, core::MultiVecDerivId fc )
        : MechanicalVisitor(mparams), fk(fk),fc(fc)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->mparams, fk.getId(mm));
        mm->resetForce(this->mparams, fc.getId(mm));
        mm->accumulateForce(this->mparams, fk.getId(mm));
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->mparams, fk.getId(mm));
        mm->resetForce(this->mparams, fc.getId(mm));
        mm->accumulateForce(this->mparams, fk.getId(mm));
        return RESULT_CONTINUE;
    }
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if( ff->isCompliance.getValue() ) ff->addForce(this->mparams, fc);
        else ff->addForce(this->mparams, fk);
        return RESULT_CONTINUE;
    }


    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        ForceMaskActivate( map->getMechFrom() );
        ForceMaskActivate( map->getMechTo() );
        map->applyJT( this->mparams, fk, fk );
        map->applyJT( this->mparams, fc, fc );
        ForceMaskDeactivate( map->getMechTo() );
    }

    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->forceMask.activate(false);
    }

    virtual void bwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
    }

    // not necessary, projection matrix is applied later in flat vector representation
//    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
//    {
//        c->projectResponse( this->mparams, fk );
//        c->projectResponse( this->mparams, fc );
//    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalComputeStiffnessAndComplianceForcesVisitor";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+fk.getName()+","+fc.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(fk);
        addWriteVector(fc);
    }
#endif
};



class MechanicalComputeStiffnessForcesAndAddingPreviousLambdasVisitor : public MechanicalVisitor
{

    core::MultiVecDerivId f,fk,fc;

public:

    MechanicalComputeStiffnessForcesAndAddingPreviousLambdasVisitor(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId f, core::MultiVecDerivId fk, core::MultiVecDerivId fc )
        : MechanicalVisitor(mparams), f(f), fk(fk),fc(fc)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->mparams, f.getId(mm));
        mm->resetForce(this->mparams, fk.getId(mm));
        mm->accumulateForce(this->mparams, fk.getId(mm));
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->mparams, f.getId(mm));
        mm->resetForce(this->mparams, fk.getId(mm));
        mm->accumulateForce(this->mparams, fk.getId(mm));
        return RESULT_CONTINUE;
    }
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if( ff->isCompliance.getValue() )
        {
            core::behavior::BaseMechanicalState* mm = ff->getContext()->getMechanicalState();
            const core::VecDerivId& fcid = fc.getId(mm);
            if( !fcid.isNull() ) // previously allocated
            {
                const core::VecDerivId& fid = f.getId(mm);
                mm->vOp( this->params, fid, fid, fcid );
            }
        }
        else
        {
            ff->addForce(this->mparams, fk);
        }
        return RESULT_CONTINUE;
    }


    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        ForceMaskActivate( map->getMechFrom() );
        ForceMaskActivate( map->getMechTo() );
        map->applyJT( this->mparams, fk, fk );
        map->applyJT( this->mparams, f, f );
        ForceMaskDeactivate( map->getMechTo() );
    }

    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->forceMask.activate(false);
    }

    virtual void bwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
    }

    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        c->projectResponse( this->mparams, fk );
        c->projectResponse( this->mparams, f );
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalComputeForcesVisitor";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+f.getName()+","+fk.getName()+","+fc.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(fk);
        addWriteVector(fc);
    }
#endif
};






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
	typedef rmat mat;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
			
    AssemblyVisitor(const core::MechanicalParams* mparams);
    virtual ~AssemblyVisitor();

//protected:
//    core::MultiVecDerivId _velId;

	// applies another visitor topDown/bottomUp based on internal
	// graph. this is to work around bogus graph traversal in case of
	// multi mappings

	void top_down(simulation::Visitor* vis) const;
	void bottom_up(simulation::Visitor* vis) const;

public:
	simulation::Node* start_node;

	// collect data chunks during visitor execution
	virtual Visitor::Result processNodeTopDown(simulation::Node* node);
	virtual void processNodeBottomUp(simulation::Node* node);
	
	// reset state
	void clear();
	
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

		// this is to remove f*cking mouse dofs
		bool mechanical; ///< is it a mechanical dof i.e. influenced by a mass or stiffness or compliance
		
		bool master() const { return mechanical && map.empty(); }
		bool compliant() const { return mechanical && C.size(); }
		
		unsigned vertex;
		
		// TODO SPtr here ?
		dofs_type* dofs;

		typedef std::map< dofs_type*, mapped> map_type;
		map_type map;
		
		// check consistency
		bool check() const;

		void debug() const;
	};

public:

	mat compliance(simulation::Node* node);
	mat proj(simulation::Node* node);
	mat odeMatrix(simulation::Node* node);
			
	chunk::map_type mapping(simulation::Node* node);
			
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

    bool isPIdentity; ///< false iff there are projective constraints

    //simulation::Node* start_node;


};

// TODO why is this here ?
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

					// note: we *DONT* filter mass/stiffness at this
					// stage since this would break direct solvers
					// (non-invertible H matrix)
					Jp = shift_right<mat>( find(offsets, vp.dofs), p->size, size_m);
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
