#ifndef SOFA_COMPONENT_MAPPING_ASSEMBLEDMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_ASSEMBLEDMULTIMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/core/MultiMapping.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/component.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include "initCompliant.h"

#include "debug.h"

namespace sofa
{

namespace component
{

namespace mapping
{


/**  
     Provides an implementation basis for assembled, sparse multi
     mappings.

     TODO: add .inl to minimize bloat / compilation times
     
     @author: maxime.tournier@inria.fr
*/

template <class TIn, class TOut >
class SOFA_Compliant_API AssembledMultiMapping : public core::MultiMapping<TIn, TOut>
{
	typedef AssembledMultiMapping self;
  public:
	SOFA_CLASS(SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut), SOFA_TEMPLATE2(core::MultiMapping,TIn,TOut));
	
	typedef core::MultiMapping<TIn, TOut> Inherit;
	typedef TIn In;
	typedef TOut Out;
	typedef typename Out::VecCoord OutVecCoord;
	typedef typename Out::VecDeriv OutVecDeriv;
	typedef typename Out::Coord OutCoord;
	typedef typename Out::Deriv OutDeriv;
	typedef typename Out::MatrixDeriv OutMatrixDeriv;
	typedef typename Out::Real Real;
	typedef typename In::Deriv InDeriv;
	typedef typename In::MatrixDeriv InMatrixDeriv;
	typedef typename In::Coord InCoord;
	typedef typename In::VecCoord InVecCoord;
	typedef typename In::VecDeriv InVecDeriv;
	typedef linearsolver::EigenSparseMatrix<TIn,TOut>  SparseMatrixEigen;

	typedef Data<OutVecCoord> OutDataVecCoord;
	typedef Data<OutVecDeriv> OutDataVecDeriv;
	typedef Data<InVecCoord> InDataVecCoord;
	typedef Data<InVecDeriv> InDataVecDeriv;


	typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
	typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;


	virtual void init() {
		assert( (this->getTo().size() == 1) && 
		        "sorry, multi mapping to multiple output dofs unimplemented" );
		
		alloc();
		
		typedef core::MultiMapping<TIn, TOut> base; // fixes g++-4.4
		base::init();
	}

	
	virtual void apply(const core::MechanicalParams*  /* PARAMS FIRST */, 
	                   const helper::vector<OutDataVecCoord*>& dataVecOutPos,
	                   const helper::vector<const InDataVecCoord*>& dataVecInPos) {
		
		unsigned n = this->getFrom().size();

		// working around non-copyable accessors yo
		in_pos_type bob(dataVecInPos[0]);
		int nn = n;
		vector<in_pos_type> in_vec(nn, bob);
		
		// i am deeply sorry for the following, but c++98's lack of move
		// semantics drives me crazy
		for( unsigned i = 0; i < n; ++i ) {
			void* addr = (void*)(&in_vec[i]);
			in_pos_type* bob = new(addr) in_pos_type(dataVecInPos[i]);
		}
		
		out_pos_type out(dataVecOutPos[0]);
		
		apply(out, in_vec);
		assemble(in_vec);
	}


	virtual void applyJ(const helper::vector<OutVecDeriv*>& outDeriv, 
	                    const helper::vector<const  InVecDeriv*>& inDeriv) {

		unsigned n = js.size();

		// working around zeroing outvecderivs (how to do that simply
		// anyways ?)
		bool first = true;
        
		for(unsigned i = 0; i < n ; ++i ) {
			if( jacobian(i).rowSize() > 0 ) {
				// small riddle lol
				if( first ) (first = false, jacobian(i).mult(*outDeriv[0], *inDeriv[i]) );
				else jacobian(i).addMult(*outDeriv[0], *inDeriv[i]);
			}
	        
		}
	}

	

	virtual void applyJT(const helper::vector< InVecDeriv*>& outDeriv, 
	                     const helper::vector<const OutVecDeriv*>& inDeriv) {
		for( unsigned i = 0, n = js.size(); i < n; ++i) {
			if( jacobian(i).rowSize() > 0 ) jacobian(i).addMultTranspose(*outDeriv[i], *inDeriv[0]);
		}
	}

	virtual void applyDJT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, 
	                      core::MultiVecDerivId /*inForce*/, 
	                      core::ConstMultiVecDerivId /*outForce*/){}

	
	virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs() {
		if( js.empty() ) std::cout << "warning: empty js for " << this->getName() << " " 
		                           <<  this->getClassName() << std::endl;

		assert( !js.empty() );
		
		return &js;
	}
 
    
  protected:
	typedef linearsolver::EigenSparseMatrix<In, Out> jacobian_type;

	
  // returns the i-th jacobian matrix
	jacobian_type& jacobian(unsigned i) {
		assert( i < js.size() );
		assert( dynamic_cast<jacobian_type*>(js[i]) );
	
		return *static_cast<jacobian_type*>(js[i]);
	}

	
	enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

	
	typedef helper::ReadAccessor< Data< typename self::InVecCoord > > in_pos_type;
	typedef helper::WriteAccessor< Data< typename self::OutVecCoord > > out_pos_type;

	// TODO pass out value as well ?
	virtual void assemble( const vector<in_pos_type>& in ) = 0;
	virtual void apply(out_pos_type& out, const vector<in_pos_type>& in ) = 0;

  private:

	// allocate jacobians
	virtual void alloc() {
		
		const unsigned n = this->getFrom().size();
		assert( n );

		// alloc
		if( js.size() != n ) {
			js.resize( n );
			
			for( unsigned i = 0; i < n; ++i ) js[i] = new SparseMatrixEigen;
		}
		
	}

	// delete jacobians
	void release() {
		for( unsigned i = 0, n = js.size(); i < n; ++i) {
			delete js[i];
			js[i] = 0;
		}
	}

	
	typedef vector< sofa::defaulttype::BaseMatrix* > js_type;
	js_type js;

	
  public:
	
	~AssembledMultiMapping() {
		release();
	}


};


}
}
}



#endif

