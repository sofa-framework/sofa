#ifndef CONTACTMULTIMAPPING_H
#define CONTACTMULTIMAPPING_H

#include "AssembledMapping.h"
#include "AssembledMultiMapping.h"

#include <Compliant/config.h>

//#include "debug.h"

#include <Compliant/utils/map.h>
#include <Compliant/utils/basis.h>
#include <Compliant/utils/nan.h>
#include <sofa/core/collision/DetectionOutput.h>

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
class ContactMultiMapping : public AssembledMultiMapping<TIn, TOut>
{
    typedef ContactMultiMapping self;
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ContactMultiMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

    typedef AssembledMultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename TIn::Real real;
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
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;

    typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;

    typedef defaulttype::Vec<2, unsigned> IndexPair;
    typedef vector< IndexPair > PairVector;
    typedef sofa::helper::vector<core::collision::DetectionOutput> DetectionOutputVector;

    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

    vector<bool> mask; ///< flag activated constraints (if empty -default- all constraints are activated)

    ContactMultiMapping()
        : contacts(NULL), pairs(NULL)
    {}

    void setDetectionOutput(DetectionOutputVector* o)
    {
        contacts = o;
    }

    void setContactPairs(PairVector* p)
    {
        pairs = p;
    }




    virtual void init()
    {
        this->getToModels()[0]->resize( pairs->size() );
        AssembledMultiMapping<TIn, TOut>::init();
    }

    virtual void reinit()
    {
        this->getToModels()[0]->resize( pairs->size() );
        AssembledMultiMapping<TIn, TOut>::reinit();
    }


    virtual void apply(typename self::out_pos_type& out,
        const vector<typename self::in_pos_type>& in) {
		
		// local frames have been computed in assemble
        assert( contacts->size() == out.size() || std::count( mask.begin(),mask.end(),true)==out.size() );
        size_t n = out.size();

		for(unsigned i = 0; i < n; ++i) {
            if( self::Nout == 2 ) // hopefully this is optimized at compilation time (known template parameter)
            {
                out[i][0] = 0;
                out[i][1] = 0;
            }
            else
            {
                out[i][0] = (real)(*contacts)[i].value;
                //out[i][0] = (TIn::getCPos( in[1] [(*pairs)[i][1]] ) - TIn::getCPos( in[0] [(*pairs)[i][0]] )) * (*contacts)[i].normal;
                if( self::Nout == 3 ) 
                {
                    out[i][1] = 0;
                    out[i][2] = 0;
                }
            }
		}
	}

protected:

    DetectionOutputVector* contacts;
    PairVector* pairs;


	virtual void assemble( const vector<typename self::in_pos_type>& in ) {


        Eigen::Matrix<real, 3, self::Nout> local_frame;

		size_t n = this->getToModels()[0]->getSize();
        
		typename Inherit::jacobian_type::CompressedMatrix& J1 = this->jacobian(0).compressedMatrix;
        typename Inherit::jacobian_type::CompressedMatrix& J2 = this->jacobian(1).compressedMatrix;
        J1.resize( (int) this->getToModels()[0]->getSize() * self::Nout, (int) this->getFromModels()[0]->getSize() * self::Nin );
        J2.resize( (int) this->getToModels()[0]->getSize() * self::Nout, (int) this->getFromModels()[1]->getSize() * self::Nin );

        J1.setZero();
        J2.setZero();

        assert( !contacts->empty() );
        assert(pairs->size() == contacts->size());
		
        for(unsigned i = 0, activatedIndex=0; i < n; ++i)
        {
            if( !mask.empty() && !mask[i] ) continue; // not activated


            if( self::Nout==2 )
            {
                Eigen::Matrix<real, 3, 1> n = utils::map( (defaulttype::Vec<3, real>)(*contacts)[i].normal );
                try{
                    local_frame.template rightCols<2>() = ker( n );
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
                local_frame.col(0) = utils::map( (defaulttype::Vec<3, real>)(*contacts)[i].normal );

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
            for( unsigned k = 0; k < self::Nout; ++k) 
            {
                unsigned row = self::Nout * activatedIndex + k;
                J1.startVec( row );
                J2.startVec( row );

                for( unsigned j = 0; j < self::Nin; ++j) {
                    real w = local_frame(j, k);
                    if(w)
                    {
                        J1.insertBack(row, (*pairs)[i][0] * Nin + j ) = -w;
                        J2.insertBack(row, (*pairs)[i][1] * Nin + j ) = w;
                    }
                }
            }

            ++activatedIndex;
        }
		J1.finalize();
        J2.finalize();
	}

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CONTACTMULTIMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec2dTypes >;
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec2fTypes >;
extern template class SOFA_Compliant_API ContactMultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
#endif
#endif

}
}
}

#endif
