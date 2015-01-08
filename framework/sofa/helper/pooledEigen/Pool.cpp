#ifdef SOFA_POOLED_SPARSE

#include "Pool.h"

#include <math.h>

static boost::pool<>* gs_pools[sizeof(size_t) * 8] = {0};

namespace Eigen { 
namespace internal {

boost::pool<>* Pool::get_pool(size_t N)
{
	const size_t n = (size_t) ceilf(logf((float)N) * (float)(1.0/M_LN2));

	if( gs_pools[n] == NULL )
		gs_pools[n] = new boost::pool<>(1 << n);

	return gs_pools[n];
}

} // namespace internal
} // namespace Eigen

#endif // SOFA_POOLED_SPARSE
