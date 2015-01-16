#ifdef SOFA_POOLED_SPARSE

#include "Pool.h"

#include <math.h>

static boost::pool<>* gs_pools[sizeof(size_t) * 8] = {0};

namespace Eigen { 
namespace internal {

inline size_t fit_pow2(size_t N)
{
	// NB: N > 0 guaranteed for our use.
#if defined(WIN32)
	unsigned long index;
	_BitScanReverse(&index, N);// __lzcnt can be unsupported and requires a check
	return index;
#elif defined(_XBOX)
	return 32 - _CountLeadingZeros(N - 1);
#else
	return (size_t) ceilf(logf((float)N) * (float)(1.0/M_LN2));
#endif
}

boost::pool<>* Pool::get_pool(size_t N)
{
	const size_t n = fit_pow2(N);

	if( gs_pools[n] == NULL )
		gs_pools[n] = new boost::pool<>(1 << n);

	return gs_pools[n];
}

} // namespace internal
} // namespace Eigen

#endif // SOFA_POOLED_SPARSE
