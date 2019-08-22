#ifndef COMPLIANT_MAPPING_PAIRINGMULTIMAPPING_H
#define COMPLIANT_MAPPING_PAIRINGMULTIMAPPING_H

#include "AssembledMultiMapping.h"
#include <Compliant/config.h>

#include "../utils/map.h"

namespace sofa {

namespace component {

namespace mapping {


/** 

    natural pairing between two vector dofs:
    
     \[ f(x, y) = x^T y \]

     use it to enforce holonomic bilateral constraints in your scene
     as: -lambda^T f(x) (+ unit compliance on end dofs)
     
     note that you need a numerical solver that can handle indefinite
     systems (e.g. minres)

     @author: Maxime Tournier

*/



template <class TIn, class TOut >
class PairingMultiMapping : public AssembledMultiMapping<TIn, TOut> {
public:
	SOFA_CLASS(SOFA_TEMPLATE2(PairingMultiMapping,TIn,TOut), 
			   SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));
	
	typedef typename TIn::Real in_real;
	typedef typename TOut::Real out_real;
	
	typedef AssembledMultiMapping<TIn, TOut> base;
	typedef PairingMultiMapping self;

    Data<SReal> sign; ///< scalar factor


    PairingMultiMapping() :
        sign(initData(&sign, (SReal)1.0, "sign", "scalar factor")) {
        
    }
    
protected:

	void apply(typename self::out_pos_type& out,
               const helper::vector< typename self::in_pos_type >& in ) {

        // auto resize output
        this->to()->resize( 1 );
        
        assert( in.size() == 2 );
        assert( in[0].size() == in[1].size() );        
        
        out_real& res = out[0][0];

        res = 0;

        for(unsigned i = 0, n = in[0].size(); i < n; ++i) {
            res += in[0][i] * in[1][i];
        }

        res *= sign.getValue();

	}



    void assemble_geometric(const helper::vector<typename self::const_in_coord_type>& /*in_pos*/,
                            const typename self::const_out_deriv_type& out_force) {
        typedef typename self::geometric_type::CompressedMatrix matrix_type;
        matrix_type& dJ = this->geometric.compressedMatrix;
        
        // out force
        const out_real& mu = out_force[0][0];

        const SReal value = sign.getValue() * mu;
        
        // matrix sizes
        const unsigned size_x = this->from(0)->getMatrixSize();
        const unsigned size_y = this->from(1)->getMatrixSize();
        assert( size_x == size_y );
        
        const unsigned size = size_x + size_y;
        
        dJ.resize( size, size );

        // we want dJ = mu * (0 I \\ I 0)

        for(unsigned i = 0; i < size_x; ++i) {
            dJ.startVec(i);
            dJ.insertBack(i, size_x + i) = value;
        }

        for(unsigned i = 0; i < size_y; ++i) {
            dJ.startVec(size_x + i);
            dJ.insertBack(size_x + i, i) = value;
        }

        dJ.finalize();

    }

    void assemble(const helper::vector< typename self::in_pos_type >& in ) {
		assert(this->getFrom()[0] != this->getFrom()[1]);

		for(unsigned i = 0, n = in.size(); i < n; ++i) {
			typename self::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;

            // resize/clean
            J.resize( 1, self::Nin * in[i].size() );

            const SReal& s = sign.getValue();
            
            const unsigned other = 1 - i;

            // fill
            J.startVec(0);
            for(unsigned j = 0, m = in[i].size(); j < m; ++j) {
                
                for(unsigned k = 0; k < self::Nin; ++k) {
                    const unsigned col = j * self::Nin + k;
                    J.insertBack(0, col) = s * in[other][j][k];
                }
            }
            J.finalize();
        }
        
	}
	


};






}
}
}

#endif
