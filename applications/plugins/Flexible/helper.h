#include <sofa/defaulttype/Mat.h>


namespace sofa
{
namespace helper
{


/// \return 0.5 * ( A + At )
template<int N, class Real>
static defaulttype::Mat<N,N,Real> symetrize( const defaulttype::Mat<N,N,Real>& A )
{
    defaulttype::Mat<N,N,Real> B;
    for( int i=0 ; i<N ; i++ )
    {
        B[i][i] = 0.5 * ( A[i][i] + A[i][i] );
        for( int j=i+1 ; j<N ; j++ )
            B[i][j] = B[j][i] = 0.5 * ( A[i][j] + A[j][i] );
    }
    return B;
}


} // namespace helper
} // namespace sofa
