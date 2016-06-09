#ifndef SOFA_COMPONENT_MAPPING_ConstantAssembledMultiMapping_H
#define SOFA_COMPONENT_MAPPING_ConstantAssembledMultiMapping_H

#include "AssembledMultiMapping.h"

namespace sofa
{

namespace component
{

namespace mapping
{


/**  
    When its parameters won't change, a MultiMapping can be *constant*
    i.e. its Jacobian and Hessian (for geometric stiffness)
    can be precomputed once for all.
     
     @author: matthieu nesme
     @date 2016
*/

template <class TIn, class TOut >
class ConstantAssembledMultiMapping : public AssembledMultiMapping<TIn, TOut>
{
    typedef ConstantAssembledMultiMapping self;
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(ConstantAssembledMultiMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

    Data<bool> d_constant; ///< If constant, the Jacobian and Hessian are build once for all during 'init' (false by default)

    virtual void init() {

        if( d_constant.getValue() )
        {
            this->alloc();

            const size_t n = this->getFrom().size();
            helper::vector<in_pos_type> in_vec; in_vec.reserve(n);
            for( unsigned i = 0; i < n; ++i ) {
                in_vec.push_back( this->getFromModels()[i]->readPositions() );
            }

            this->assemble( in_vec );
            assemble_hessian( in_vec );
        }

        Inherit1::init();
	}


    typedef typename Inherit1::OutDataVecCoord OutDataVecCoord;
    typedef typename Inherit1::InDataVecCoord InDataVecCoord;
	
    virtual void apply(const core::MechanicalParams* mparams,
	                   const helper::vector<OutDataVecCoord*>& dataVecOutPos,
	                   const helper::vector<const InDataVecCoord*>& dataVecInPos) {
        if( !d_constant.getValue() )
            return Inherit1::apply( mparams, dataVecOutPos, dataVecInPos );
	

		const unsigned n = this->getFrom().size();

        helper::vector<in_pos_type> in_vec; in_vec.reserve(n);

		for( unsigned i = 0; i < n; ++i ) {
			in_vec.push_back( in_pos_type(dataVecInPos[i]) );
		}
		
		out_pos_type out(dataVecOutPos[0]);
		
        apply(out, in_vec);
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

        if( !d_constant.getValue() ) assemble_hessian( in_vec );
        this->assemble_geometric(in_vec, out_force);
    }

	
protected:

    typedef typename Inherit1::in_pos_type in_pos_type;
    typedef typename Inherit1::out_pos_type out_pos_type;
    typedef typename Inherit1::const_in_coord_type const_in_coord_type;
    typedef typename Inherit1::const_out_deriv_type const_out_deriv_type;


    ConstantAssembledMultiMapping()
        : d_constant( initData(&d_constant, false, "constant", "Can the Jacobian and Hessian be precomputed?") )
    {}


    // to remove some warnings
    virtual void apply( out_pos_type& out, const helper::vector<in_pos_type>& in ) = 0;


    /// The Hessian can be constant, while geometric stiffness is not
    /// K = Hessian * out_force
    /// In a constant mapping, the Hessian can be precomputed during 'init'
    /// and then be used in 'assemble_geometric' to perform Hessian * out_force.
    /// Note the Hessian type is not trivial and the user can choose what is best in his case.
    virtual void assemble_hessian( const helper::vector<const_in_coord_type>& /*in*/ ) {}
};


}
}
}



#endif

