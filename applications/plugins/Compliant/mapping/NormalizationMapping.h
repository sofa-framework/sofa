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
    Data<Indices> d_indices; ///< indices of vector to normalize

	
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
            const Coord& v = in[index];


            Real t1 = v[2] * v[2];
            Real t2 = v[0] * v[0];
            Real t3 = v[1] * v[1];
            Real t4 = t1 + t2 + t3;
            Real t5 = 1. / ( t4 * sqrt(t4) ); //pow(t4, -0.3e1 / 0.2e1);
            t4 = t4 * t5;
            Real t6 = v[1] * t5 * v[0];
            Real t7 = v[2] * t5;
            Real t8 = t7 * v[0];
            t7 = t7 * v[1];

            J.startVec( k*N );
            J.insertBack(k*N  , index*N   ) = -t2 * t5 + t4;
            J.insertBack(k*N  , index*N+1 ) = -t6;
            J.insertBack(k*N  , index*N+2 ) = -t8;

            J.startVec( k*N+1 );
            J.insertBack(k*N+1, index*N   ) = -t6;
            J.insertBack(k*N+1, index*N+1 ) = -t3 * t5 + t4;
            J.insertBack(k*N+1, index*N+2 ) = -t7;

            J.startVec( k*N+2 );
            J.insertBack(k*N+2, index*N   ) = -t8;
            J.insertBack(k*N+2, index*N+1 ) = -t7;
            J.insertBack(k*N+2, index*N+2 ) = -t1 * t5 + t4;
        }
        J.finalize();
	}
};



}
}
}


#endif
