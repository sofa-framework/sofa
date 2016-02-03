#ifndef SOFA_COMPONENT_MAPPING_ASSEMBLEDMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_ASSEMBLEDMULTIMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/core/MultiMapping.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <Compliant/config.h>

// #include "debug.h"

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
class AssembledMultiMapping : public core::MultiMapping<TIn, TOut>
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
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
	typedef typename In::Deriv InDeriv;
	typedef typename In::MatrixDeriv InMatrixDeriv;
	typedef typename In::Coord InCoord;
	typedef typename In::VecCoord InVecCoord;
	typedef typename In::VecDeriv InVecDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
	typedef linearsolver::EigenSparseMatrix<TIn,TOut>  SparseMatrixEigen;

	typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
	typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;


	virtual void init() {
		assert( (this->getTo().size() == 1) && 
		        "sorry, multi mapping to multiple output dofs unimplemented" );
		
        Inherit::init();
	}

    virtual void reinit() {

        Inherit::apply(core::MechanicalParams::defaultInstance() , core::VecCoordId::position(), core::ConstVecCoordId::position());
        Inherit::applyJ(core::MechanicalParams::defaultInstance() , core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
        if (this->f_applyRestPosition.getValue())
            Inherit::apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::restPosition(), core::ConstVecCoordId::restPosition());

        Inherit::reinit();
    }

    typedef linearsolver::EigenSparseMatrix<In, In> geometric_type;
    geometric_type geometric;
    
    virtual const defaulttype::BaseMatrix* getK() {
        if( geometric.compressedMatrix.nonZeros() ) return &geometric;
        else return NULL;
    }


    virtual void updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId force ) {

		const unsigned n = this->getFrom().size();

        helper::vector<const_in_coord_type> in_vec; in_vec.reserve(n);

        core::ConstMultiVecCoordId pos = core::ConstVecCoordId::position();
        
		for( unsigned i = 0; i < n; ++i ) {
            const core::State<TIn>* from = this->getFromModels()[i];
            const_in_coord_type in_pos( *pos[from].read() );
			in_vec.push_back( in_pos );
		}

        const core::State<TOut>* to = this->getToModels()[0];
        const_out_deriv_type out_force( *force[to].read() );

        this->assemble_geometric(in_vec, out_force);
    }

	
	virtual void apply(const core::MechanicalParams* , 
	                   const helper::vector<OutDataVecCoord*>& dataVecOutPos,
	                   const helper::vector<const InDataVecCoord*>& dataVecInPos) {
		alloc();
	
		const unsigned n = this->getFrom().size();

        helper::vector<in_pos_type> in_vec; in_vec.reserve(n);

		for( unsigned i = 0; i < n; ++i ) {
			in_vec.push_back( in_pos_type(dataVecInPos[i]) );
		}
		
		out_pos_type out(dataVecOutPos[0]);
		
		apply(out, in_vec);
		assemble(in_vec);
	}



    virtual void applyJ(const core::MechanicalParams*, const helper::vector<OutDataVecDeriv*>& outDeriv, const helper::vector<const InDataVecDeriv*>& inDeriv)
    {
        unsigned n = js.size();
        unsigned i = 0;

        // let the first valid jacobian set its contribution    out = J_0 * in_0
        for( ; i < n ; ++i )
        {
            if( jacobian(i).rowSize() > 0 )
            {
                jacobian(i).mult(*outDeriv[0], *inDeriv[i]);
                break;
            }
        }

        ++i;

        // the next valid jacobians will add their contributions    out += J_i * in_i
        for( ; i < n ; ++i )
        {
            if( jacobian(i).rowSize() > 0 )
                jacobian(i).addMult(*outDeriv[0], *inDeriv[i]);
        }

    }

    void debug() {
		std::cerr << this->getClassName() << std::endl;
		for( unsigned i = 0, n = js.size(); i < n; ++i) {
			std::cerr << "from: " <<  this->getFrom()[i]->getContext()->getName() 
					  << "/" << this->getFrom()[i]->getName() << std::endl;
		}
		std::cerr << "to: " << this->getTo()[0]->getContext()->getName() << "/" << this->getTo()[0]->getName() << std::endl;
		std::cerr << std::endl;
	}

	virtual void applyJT(const core::MechanicalParams*,
						 const helper::vector< InDataVecDeriv*>& outDeriv, 
	                     const helper::vector<const OutDataVecDeriv*>& inDeriv) {
		for( unsigned i = 0, n = js.size(); i < n; ++i) {
			if( jacobian(i).rowSize() > 0 ) {
				jacobian(i).addMultTranspose(*outDeriv[i], *inDeriv[0]);
			}
		}
	}

    virtual void applyDJT(const core::MechanicalParams*,
	                      core::MultiVecDerivId /*inForce*/, 
                          core::ConstMultiVecDerivId /*outForce*/)
    {
        if( geometric.compressedMatrix.nonZeros() ) serr<<"applyDJT is not yet implemented"<<sendl;
        // TODO implement it!
    }

    virtual void applyJT( const core::ConstraintParams*,
						  const helper::vector< typename self::InDataMatrixDeriv* >& , 
						  const helper::vector< const typename self::OutDataMatrixDeriv* >&  ) {
		// throw std::logic_error("not implemented");
	}

	
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() {
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


    // let's do this once and for all
    typedef helper::ReadAccessor< Data< typename self::InVecCoord > > const_in_coord_type;
    typedef helper::WriteAccessor< Data< typename self::InVecCoord > > in_coord_type;

    typedef helper::ReadAccessor< Data< typename self::OutVecCoord > > const_out_coord_type;
    typedef helper::WriteAccessor< Data< typename self::OutVecCoord > > out_coord_type;

    typedef helper::ReadAccessor< Data< typename self::InVecDeriv > > const_in_deriv_type;
    typedef helper::WriteAccessor< Data< typename self::InVecDeriv > > in_deriv_type;
    
    typedef helper::ReadAccessor< Data< typename self::OutVecDeriv > > const_out_deriv_type;
    typedef helper::WriteAccessor< Data< typename self::OutVecDeriv > > out_deriv_type;


	// TODO rename in_coord_type/out_coord_type
    typedef helper::ReadAccessor< Data< typename self::InVecCoord > > in_pos_type;
    typedef helper::WriteOnlyAccessor< Data< typename self::OutVecCoord > > out_pos_type;

    typedef helper::ReadAccessor< Data< typename self::InVecDeriv > > in_vel_type;
    typedef helper::WriteAccessor< Data< typename self::OutVecDeriv > > out_vel_type;

    typedef helper::ReadAccessor< Data< typename self::OutVecDeriv > > in_force_type;
    typedef helper::WriteAccessor< Data< typename self::InVecDeriv > > out_force_type;
	
	
	// perform a jacobian blocs assembly
	// TODO pass out value as well ?
    virtual void assemble( const helper::vector<in_pos_type>& in ) = 0;

    virtual void assemble_geometric( const helper::vector<const_in_coord_type>& /*in*/,
                                     const const_out_deriv_type& /*out*/) { }
    
    using Inherit::apply;
	// perform mapping operation on positions
    virtual void apply(out_pos_type& out, 
                       const helper::vector<in_pos_type>& in ) = 0;

  private:

	// allocate jacobians
	virtual void alloc() {
		
		const unsigned n = this->getFrom().size();
		if( n != js.size() ) {
            release();

            // alloc
            if( js.size() != n ) {
                js.resize( n );

                for( unsigned i = 0; i < n; ++i ) js[i] = new SparseMatrixEigen;
            }
		}
	}

	// delete jacobians
    void release() {
        for( unsigned i = 0, n = js.size(); i < n; ++i) {
			delete js[i];
			js[i] = 0;
		}
	}

	
    typedef helper::vector< sofa::defaulttype::BaseMatrix* > js_type;
	js_type js;


  protected:
	
	core::behavior::BaseMechanicalState* from(unsigned i) {
		// TODO assert
        return this->getFrom()[i]->toBaseMechanicalState();
	}

	core::behavior::BaseMechanicalState* to(unsigned i = 0) {
		// TODO assert
        return this->getTo()[i]->toBaseMechanicalState();
	}

	
  public:
	
	~AssembledMultiMapping() {
		release();
	}


};


}
}
}



#endif

