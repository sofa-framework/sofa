#ifndef NormalizationMAPPING_H
#define NormalizationMAPPING_H

#include <Compliant/config.h>

#include "AssembledMapping.h"
#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Maps a 3d vector to its normalization:

 v ->  v / ||v||

 @author Matthieu Nesme
 @date 2016



dn_i/dv_i = (v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(3/2)
dn_i/dv_j = - v_i*v_j / (v_i^2+v_j^2+v_k^2)^(3/2)


d2n_i/d2v_i = - 3.v_i.(v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
d2n_i/dv_idv_j = d2n_i/dv_jdv_i = - v_j.(-2.v_i^2+v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
d2n_i/d2v_j = - v_i.(v_i^2-2.v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
d2n_i/dv_jdv_k = 3*a*b*c / (v_i^2+v_j^2+v_k^2)^(5/2)

*/
template <class T >
class SOFA_Compliant_API NormalizationMapping : public AssembledMapping<T, T>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE(NormalizationMapping,T), SOFA_TEMPLATE2(AssembledMapping,T,T));
	
    typedef NormalizationMapping<T> self;
    typedef typename T::Coord Coord;
    typedef typename T::Real Real;

    typedef helper::vector<unsigned> Indices;
    Data<Indices> d_indices;

	
    NormalizationMapping()
        : d_indices( initData(&d_indices, "indices", "indices of vector to normalize") )
    {}

    enum {N = T::deriv_total_size };

    virtual void init()
    {
        reinit();
        Inherit1::init();
    }

    virtual void reinit()
    {
        helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(d_indices);
        size_t size = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

        this->getToModel()->resize( size );
        Inherit1::reinit();
    }

	virtual void apply(typename self::out_pos_type& out, 
	                   const typename self::in_pos_type& in )  {

        helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(d_indices);
        size_t size = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

        assert( out.size() == size );

        for( size_t j = 0; j < size; ++j)
        {
            const unsigned& index = indices.empty() ? j : indices[j] ;
            out[j] = in[index].normalized(); // todo keep norm computation for Jacobian and Hessian
        }
	}

	virtual void assemble( const typename self::in_pos_type& in ) {

        helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(d_indices);
        size_t size = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

        typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        this->jacobian.resizeBlocks( size, in.size() );
        J.reserve(size*N*N);

        for(size_t k = 0; k < size; ++k)
        {
            const unsigned& index = indices.empty() ? k : indices[k] ;
            const Coord& v = in[index]; // [a,b,c]

            const Real a2 = v[0] * v[0];
            const Real b2 = v[1] * v[1];
            const Real c2 = v[2] * v[2];

            Real denom = a2+b2+c2;
            denom = 1. / ( denom * sqrt(denom) ); // (a2+b2+c2)^(-3/2)

            const Real ab = -v[0]*v[1]*denom;
            const Real ac = -v[0]*v[2]*denom;
            const Real bc = -v[1]*v[2]*denom;

            size_t row = k*N;
            const size_t col = index*N;

            J.startVec( row );
            J.insertBack(row, col   ) = (b2+c2)*denom; // dn_0/dv_0
            J.insertBack(row, col+1 ) = ab;            // dn_1/dv_0
            J.insertBack(row, col+2 ) = ac;            // dn_2/dv_0

            J.startVec( ++row );
            J.insertBack(row, col   ) = ab;
            J.insertBack(row, col+1 ) = (a2+c2)*denom;
            J.insertBack(row, col+2 ) = bc;

            J.startVec( ++row );
            J.insertBack(row, col   ) = ac;
            J.insertBack(row, col+1 ) = bc;
            J.insertBack(row, col+2 ) = (a2+b2)*denom;
        }
        J.finalize();
	}


    virtual void assemble_geometric( const typename self::in_pos_type& in, const typename self::out_force_type& out )
    {
        helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(d_indices);
        size_t size = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

        typedef defaulttype::Mat<3,3,Real> HessianBlock[3]; // 3x3x3 tensor
        typedef defaulttype::Mat<3,3,Real> Block;

        typename self::geometric_type::CompressedMatrix& K = this->geometric.compressedMatrix;
        this->geometric.resizeBlocks( in.size(), in.size() );
        K.reserve(in.size()*N*N);

        // temp
        HessianBlock H;

        for(size_t i = 0; i < size; ++i)
        {
            const unsigned& index = indices.empty() ? i : indices[i] ;

            const Coord& childForce = out[i];

            const Coord& v = in[index];
            const Coord& v2 = Coord( v[0]*v[0], v[1]*v[1], v[2]*v[2] );

            const Real sum2 = v2[0]+v2[1]+v2[2];
            const Real denom = 1. / pow( sum2, 5./2. );

            const Real prod = 3*v[0]*v[1]*v[2]*denom;



            // dn/dv
            for( int l=0 ; l<N ; ++l ) // dn
            {
                for( int j=0 ; j<N ; ++j ) // dv_i
                {
                    for( int k=0 ; k<N ; ++k ) // dv_j
                    {
                        if( l==j && l==k ) H[l][j][k] = -3 * v[l] * ( sum2 - v2[l] ) * denom;  // d2n_i/d2v_i = - 3.v_i.(v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
                        else if( l==j ) H[l][j][k] = - v[k] * ( sum2 - 3*v2[l] ) * denom; // d2n_i/dv_idv_j = d2n_i/dv_jdv_i = - v_j.(-2.v_i^2+v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
                        else if( l==k ) H[l][j][k] = - v[j] * ( sum2 - 3*v2[l] ) * denom; // d2n_i/dv_idv_j = d2n_i/dv_jdv_i = - v_j.(-2.v_i^2+v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
                        else if( j==k ) H[l][j][k] = -v[l] * ( sum2 - 3*v2[j] ) * denom; // d2n_i/d2v_j = - v_i.(v_i^2-2.v_j^2+v_k^2) / (v_i^2+v_j^2+v_k^2)^(5/2)
                        else H[l][j][k] = prod; // d2n_i/dv_jdv_k = 3*a*b*c / (v_i^2+v_j^2+v_k^2)^(5/2)
                    }
                }
            }


            Block b;
            for( int l=0 ; l<N ; ++l ) // dn
            {
                for( int j=0 ; j<N ; ++j ) // dv_i
                {
                    for( int k=0 ; k<N ; ++k ) // dv_j
                    {
                        b[j][k] += H[l][j][k] * childForce[l];
                    }
                }
            }


            for( int j=0 ; j<N ; ++j ) // dv_i
            {
                const size_t row = index*N+j;
                K.startVec( row );
                for( int k=0 ; k<N ; ++k ) // dv_j
                {
                    const size_t col = index*N+k;

                    K.insertBack(row, col) = b[j][k];
                }
            }

        }
        K.finalize();
    }

};



}
}
}


#endif
