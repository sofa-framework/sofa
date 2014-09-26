#ifndef EIGEN_POOL_H
#define EIGEN_POOL_H

#ifndef BOOST_CHRONO_HEADER_ONLY
	#define BOOST_CHRONO_HEADER_ONLY
	#include <boost/pool/pool.hpp>
	#undef BOOST_CHRONO_HEADER_ONLY
#else
	#include <boost/pool/pool.hpp>
#endif

namespace Eigen { 
namespace internal {

namespace {
/// Validity checker. This class sets its boolean to false when
/// destroyed.  It is used to monitor the destruction of static
/// members.
struct ValidityChecker
{
	bool isValid;
	ValidityChecker() : isValid( true ) {}

	~ValidityChecker() { isValid = false; }
};
}

/// Pool for the matrices. We create several pools for different size of matrices.
class Pool
{
public:
	/// Allocate in the pool if n <= 1024. Otherwise, malloc directly
	/// in the system memory.
	template< typename T > 
	static T * allocate( const size_t n )
	{

	  boost::pool<> *pool = get< T >(n);
	  void* buffer = 0;
	  if ( pool )
	  {
		buffer = pool->malloc();
	  }

	  else
	  {
		buffer = ::malloc( n * sizeof( T ) );
		return reinterpret_cast< T* >( buffer );
	  }


	  if( !buffer )
	  {
		std::cout << "Allocation failed." << std::endl;
	  }

	  return reinterpret_cast< T* >( buffer );
	}

	/// Reallocates the given buffer with the given size.
	template< typename T >
	static T * reallocate( T * buffer, const size_t n )
	{
		/// Retrieves the storage currently responsible for the buffer.
		boost::pool<> * oldStorage = get( buffer );

		/// If the buffer has been allocated in a storage,
		/// we will manage the reallocation.
		if( oldStorage )
		{
			/// If the new size feets into the allocated buffer,
			/// we simply return the same buffer.
			if( oldStorage->get_requested_size() >= n )
			{
				return buffer;
			}
			/// Else, we get the new storage, allocate a new buffer, copy the data and free the previous buffer.
			else
			{
				T * newBuffer = allocate< T >( n );

				std::memcpy( newBuffer, buffer, oldStorage->requested_size() * sizeof( T ) );
				oldStorage->free( buffer );

				return newBuffer;
			}
		}
		/// The buffer has not been allocated in a storage,
		/// so we do a standard rellocation.
		else
		{
			void * result = ::realloc(buffer, n * sizeof( T ) );
			return reinterpret_cast< T* >( result );
		}
	}

	/// Free in the pool if n <= 4096. Otherwise, it is system memory,
	/// free it there.
	template< typename T >
	static void free( T * buffer )
	{
	  if( !buffer )
		return;

	  /// Get the pool that has been used to allocate the buffer,
	  /// and free the memory.
	  boost::pool<> * pool = get( buffer );
	  if( pool != 0 )
	  {
		  pool->free( buffer );
	  }
	  /// Else, the buffer has been allocated outised a pool
	  /// and it will be freed directly.
	  else
	  {
		::free( buffer );
	  }
	}

	/// Get the pool. There are small matrices and big one. To prevent
	/// to many memory usage for small matrices, we create pools of
	/// different sizes. This could be tuned.
	template< typename T>
	static boost::pool<> * get( int n )
	{
	  if( n <= 32 / sizeof( T ) )
	  {
		return get< 32 >();
	  }	
	  if( n <= 256 / sizeof( T ) )
	  {
		return get< 256 >();
	  }	
	  if( n <= 2048 / sizeof( T ) )
	  {
		return get< 2048 >();
	  }	
	  if( n <= 4096 / sizeof( T ) )
	  {
		return get< 4096 >();
	  } 
	  return 0;
	}

	/// Get the pool that owns the given buffer.
	static boost::pool<> * get( void * buffer )
	{
	  boost::pool<> * pool = 0;
	  
	  pool = get< 32 >();
	  if( pool->is_from(buffer) )
	  {
		  return pool;
	  }

	  pool = get< 256 >();
	  if( pool->is_from(buffer) )
	  {
		  return pool;
	  }

	  pool = get< 2048 >();
	  if( pool->is_from(buffer) )
	  {
		  return pool;
	  }

	  pool = get< 4096 >();
	  if( pool->is_from(buffer) )
	  {
		  return pool;
	  }

	  return 0;
	}


	// Creation of the pools. Pools are destroyed after main. The
	// validity checker detects that destruction and in that case a
	// null pointer is returned.
	//
	// If matrices can be used after main, we should not allow the
	// pool to be destroyed (use pointer or boost::singleton_pool),
	// but there will be memory leaks.
	template< int N >
	static boost::pool<> * get()
	{
	  static boost::pool<> storage( N );
	  static ValidityChecker vc;

	  if ( vc.isValid )
		return &storage;
	  else
		return 0;
	}




};


} // namespace internal
} // namespace Eigen

#endif // EIGEN_POOL_H
