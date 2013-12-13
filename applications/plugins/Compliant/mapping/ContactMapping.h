#ifndef CONTACTMAPPING_H
#define CONTACTMAPPING_H

#include "AssembledMapping.h"

#include "../initCompliant.h"

//#include "debug.h"

#include "utils/map.h"
#include "utils/basis.h"
#include "utils/nan.h"

#include <limits>

namespace sofa
{

namespace component
{

namespace mapping
{

// maps relative positions to a contact frame (n, nT1, nT2)

// @penetrations are given (the ones computed during the collision detection and so signed), they are directly used as the normal components.

// depending on TOut dimension (3 or 1), tangent components will be available or not

// If @normals are given, these directions are used as the first axis of the local contact frames.
// If the normal is invalid or not given, the first direction is based on the normalized dof value.

// author: maxime.tournier@inria.fr

template <class TIn, class TOut >
class ContactMapping : public AssembledMapping<TIn, TOut>
{
	typedef ContactMapping self;
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ContactMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

	typedef typename TIn::Real real;

	typedef vector< defaulttype::Vec<3, real> > normal_type;
	normal_type normal;

	typedef vector< real > penetration_type;
	penetration_type penetrations;

	
protected:

	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in) {
		
		// local frames have been computed in assemble

		assert( in.size() == out.size() );
        assert( in.size() == penetrations.size() );

		unsigned n = in.size();

		for(unsigned i = 0; i < n; ++i) {

            out[i][0] = penetrations[i];

//             std::cout << SOFA_CLASS_METHOD<<"normal " << normal[i] << std::endl;
//             std::cout << SOFA_CLASS_METHOD<< "penetration " << penetrations[i] << " "<< out[i][0]<< std::endl;
			
			if( self::Nout == 3 ) {
				out[i][1] = 0;
				out[i][2] = 0;
			}
			
		}

	}


	virtual void assemble( const typename self::in_pos_type& in_pos ) {


		Eigen::Matrix<real, 3, self::Nout> local_frame;

		unsigned n = in_pos.size();
		
		// this->jacobian.resizeBlocks(d.size(), in_pos.size() );
		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
		J.resize( n * self::Nout, n * self::Nin );
		
		J.setZero();
		
		for(unsigned i = 0; i < n; ++i)
			{
				assert( !normal.empty() );
				assert( std::abs( normal[i].norm() - 1 ) <= std::numeric_limits<SReal>::epsilon() );
				
				// first vector is normal
                local_frame.col(0) = ::map( normal[i] );
				
				// possibly tangent directions
				if( self::Nout == 3 ) {
					Eigen::Matrix<real, 3, 1> n = local_frame.col(0);
					local_frame.template rightCols<2>() = ker( n );
				}
				
				// make sure we're cool
				assert( !has_nan(local_frame) );
				
				// rows
				for( unsigned k = 0; k < self::Nout; ++k) {
					unsigned row = self::Nout * i + k;

					J.startVec( row );
					// cols
					for( unsigned j = 0; j < self::Nin; ++j) {
						unsigned col = self::Nin * i + j;

						// local_frame transpose
						J.insertBack(row, col) = local_frame(j, k);
					}

				}
			}

		J.finalize();

	}

};
	

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_ContactMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
#endif
#endif

}
}
}

#endif
