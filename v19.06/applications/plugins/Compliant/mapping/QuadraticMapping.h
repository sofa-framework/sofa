#ifndef COMPLIANT_MAPPING_QUADRATICMAPPING_H
#define COMPLIANT_MAPPING_QUADRATICMAPPING_H


#include <Compliant/config.h>
#include "AssembledMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
   Quadratic mapping: x -> 1/2 k ||x||^2
*/


template <class TIn, class TOut >
class SOFA_Compliant_API QuadraticMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(QuadraticMapping,TIn,TOut),
               SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
    typedef QuadraticMapping self;
    typedef typename TOut::Real out_real;
    
    Data< SReal > stiffness; ///< scalar factor
    
    QuadraticMapping()
        : stiffness( initData(&stiffness, (SReal)1.0, "stiffness", "scalar factor") ) {
        assert( self::Nout == 1 );
    }

    
    virtual void apply(typename self::out_pos_type& out,
                       const typename self::in_pos_type& in ) {
        
        // automatic output resize
        this->getToModel()->resize( 1 );
        
        out_real& res = out[0][0];

        res = 0;

        for(unsigned i = 0, n = in.size(); i < n; ++i) {
            res += in[i] * in[i];
        }

        res *= stiffness.getValue() / 2;
	}


    virtual void assemble( const typename self::in_pos_type& in ) {
        typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        J.resize( 1, self::Nin * in.size());
        J.reserve( self::Nin * in.size() );

        J.startVec( 0 );

        const SReal& s = stiffness.getValue();
        
        for(unsigned i = 0, n = in.size(); i < n; ++i) {

            for(unsigned j = 0; j < self::Nin; ++j) {
                const unsigned col = i * self::Nin + j;
                J.insertBack(0, col) = s * in[i][j];
            }
        }

        J.finalize();
        
	}


    virtual void assemble_geometric( const typename self::in_pos_type& in,
                                     const typename self::out_force_type& out) {
        const out_real& mu = out[0][0];
        const SReal& s = stiffness.getValue();

        const SReal value = s * mu;
        
        typename self::geometric_type::CompressedMatrix& dJ = this->geometric.compressedMatrix;
        const unsigned size = self::Nin * in.size();
        dJ.resize( size, size );
        dJ.reserve( size );

        for(unsigned i = 0, n = in.size(); i < n; ++i) {
            dJ.startVec(i);
            dJ.insertBack(i, i) = value;
        }

        dJ.finalize();        
    }

	
};




}
}
}


#endif
