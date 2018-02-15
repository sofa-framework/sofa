#ifndef CONTACTMAPPING_H
#define CONTACTMAPPING_H

#include "AssembledMapping.h"

#include <Compliant/config.h>

//#include "debug.h"

#include <Compliant/utils/map.h>
#include <Compliant/utils/basis.h>
#include <Compliant/utils/nan.h>

#include <limits>

namespace sofa
{

namespace component
{

namespace mapping
{

// maps relative positions to a contact frame
// 1D -> (n)
// 2D -> (nT1, nT2)
// 3D -> (n, nT1, nT2)


// depending on TOut dimension (3 or 1), tangent components will be available or not
// special case for TOut==2, only the tangent components are available (but not the normal component)

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

    typedef helper::vector< defaulttype::Vec<3, real> > normal_type;
    Data<normal_type> normal; ///< contact normals

    helper::vector<bool> mask; ///< flag activated constraints (if empty -default- all constraints are activated)


    ContactMapping()
        : normal(initData(&normal, "normal", "contact normals"))
    {}

protected:

	virtual void apply(typename self::out_pos_type& out,
                       const typename self::in_pos_type& in) {
		
		// local frames have been computed in assemble

        (void)in;
        assert( in.size() == out.size() || (size_t)std::count( mask.begin(),mask.end(),true)==out.size() );

        unsigned n = out.size();

		for(unsigned i = 0; i < n; ++i) {

            if( self::Nout == 2 ) // hopefully this is optimized at compilation time (known template parameter)
            {
                out[i][0] = 0;
                out[i][1] = 0;
            }
            else
            {

                // out[i][0] = penetrations are given during contact creation, they are directly used as the normal components.
                // (the ones computed during the collision detection and so signed)

                if( self::Nout == 3 ) {
                    out[i][1] = 0;
                    out[i][2] = 0;
                }
            }
			
		}

	}


	virtual void assemble( const typename self::in_pos_type& in_pos ) {

        Eigen::Matrix<real, 3, self::Nout> local_frame;

		unsigned n = in_pos.size();

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        J.resize( this->toModel->getSize() * self::Nout, n * self::Nin );
        J.reserve( n * self::Nout * self::Nin );

        assert( !normal.getValue().empty() && normal.getValue().size() == n );
		
        for(unsigned i = 0, activatedIndex=0; i < n; ++i)
        {
//          assert( std::abs( normal[i].norm() - 1 ) <= std::numeric_limits<SReal>::epsilon() );


            if( !mask.empty() && !mask[i] ) continue; // not activated


            if( self::Nout==2 )
            {
                Eigen::Matrix<real, 3, 1> n = utils::map( normal.getValue()[i] );
                try{
                    local_frame.template leftCols<2>() = ker( n );
                }
                catch( const std::logic_error& ) {
                    std::cout << "skipping degenerate normal for contact " << i
                            << ": " << n.transpose() << std::endl;
                    local_frame.setZero();
                }
            }
            else
            {
                // first vector is normal
                local_frame.col(0) = utils::map( normal.getValue()[i] );

                // possibly tangent directions
                if( self::Nout == 3 ) {
                    Eigen::Matrix<real, 3, 1> n = local_frame.col(0);
                    try{
                      local_frame.template rightCols<2>() = ker( n );
                    }
                    catch( const std::logic_error& ) {
                      std::cout << "skipping degenerate normal for contact " << i
                                << ": " << n.transpose() << std::endl;
                      local_frame.setZero();
                    }
                }
            }

            // make sure we're cool
            assert( !has_nan(local_frame) );

            // rows
            for( unsigned k = 0; k < self::Nout; ++k) {
                unsigned row = self::Nout * activatedIndex + k;

                J.startVec( row );
                // cols
                for( unsigned j = 0; j < self::Nin; ++j) {
                    unsigned col = self::Nin * i + j;

                    // local_frame transpose
                    real w = local_frame(j, k);
                    if(w) J.insertBack(row, col) = w;
                }

            }
            ++activatedIndex;
        }

		J.finalize();

	}

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_ContactMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3dTypes, defaulttype::Vec2dTypes >;
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3fTypes, defaulttype::Vec2fTypes >;
extern template class SOFA_Compliant_API ContactMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
#endif
#endif

}
}
}

#endif
