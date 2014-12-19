/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2014 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_SPTRPOOL_H
#define SOFA_CORE_OBJECTMODEL_SPTRPOOL_H

#include <sofa/core/objectmodel/SPtr.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 * \brief Object pool (for Base derived objects managed through SPtr).
 *
 * For each type for which you want to use a pool, you need to:
 *
 * - use the SOFA_POOLABLE_CLASS macro instead of SOFA_CLASS.
 * - preferably, export the matching Pool<T> implementation (if you don't it will still work, 
 *   but each module instancing the type will have a separate pool).
 * - override the virtual recycle() method in order to reset the state of your object to something compatible with reuse
 *   (you should at least release all references to external objects so that they are released to their pool,
 *   but it may not be necessary to reset all internal variables if you know they will be overriden anyway before being read)
 * - if the type takes constructors with parameters, implement the construct(...) method with equivalent parameters,
 *   which will be called instead of the normal constructor when an object is reused from the pool.
 *
 *   @note construct(...) should be protected and non virtual like the constructor, this way you will see if you forgot to
 *         write the method instead of transparently calling the equivalent method on the parent class.
 *   @note there is no need for an empty construct() method, the recycle() method is responsible to set a state equivalent 
 *         to having been created with the empty constructor.
 */
template<typename T>
class Pool
{
public:
	/// Expand the pool of unused objects of a type in advance.
	/// @note This is not like a "std::vector::reserve": expanding twice with the same quantity will allocate new objects.
	static void expand(size_t count);

	/// Removes all available objects from the the pool, freeing them immediately.
	/// @note objects currently allocated from the pool will still be released into it once they lose their last reference.
	/// @note this is not really usefull while the program is running, but can be used to release memory at the end of the program
	///       sooner than when the static pool object will be released. It's also useful for tests, to get the pool to a known state.
	static void clear();

	/// Gets the current number of objects allocated (ie. live) from this pool.
	static int getAllocatedCount() { return instance().allocated; }

	/// Gets the current number of objects available for reuse in this pool.
	static int getAvailableCount() { return instance().store.size(); }

private:
	friend class BaseNew<T, true>;
	friend typename T;
	typedef helper::vector<T*> PoolVector;
	class PoolLock 
	{
	public:
		PoolLock() {}
		~PoolLock() {}
	}; // TODO: In multi-threaded mode, this should lock a mutex guarding access to the pool in the constructor, and unlock it in the destructor.

	static T* allocate();
	static void release(T*);
	static Pool& instance();
	Pool();
	~Pool();

	PoolVector store;
	int allocated;
};

/// Major operations and settings on how to manipulate a pooled object are defined
/// through a policy in case you need to implement them differently for a given type.
template<class T>
struct PoolPolicy 
{
	enum { ReserveCount = 128 }; 
	static void recycle(T* obj) { static_cast<Base*>(obj)->recycle(); }
	static bool isPooled(const T* obj) { return obj->isPooled(); }
	static void setPooled(T* obj, bool pooled) { return obj->setPooled(pooled); }
};

/// Global pooling settings
class SOFA_CORE_API PoolSettings
{
public:
	/// Enable allocation pooling for NewFromPool calls (this only affects the current thread).
	static void enable(bool enabled);
	/// Test if allocation pooling is enabled for the current thread.
	static bool isEnabled();
};


template<class T>
void Pool<T>::expand(size_t count)
{
	PoolLock l;

	Pool& pool = instance();
	size_t poolSize = pool.store.size();
	pool.store.resize(poolSize + count);
	for (size_t i = poolSize, n = poolSize + count; i < n; ++i)
	{
		T* result = BaseNew<T,true>::internal_allocate();
		PoolPolicy<T>::setPooled(result, true);
		pool.store[i] = result;
	}
}

template<class T>
void Pool<T>::clear()
{
	PoolLock l;

	Pool& pool = instance();

	if (!pool.store.empty())
	{
		for (PoolVector::iterator it = pool.store.begin(), end = pool.store.end(); it != end; ++it)
			BaseNew<T,true>::internal_release(*it);
		std::swap(pool.store, PoolVector());
		pool.store.reserve(PoolPolicy<T>::ReserveCount);
	}
}

template<class T>
T* Pool<T>::allocate()
{
	if (PoolSettings::isEnabled())
	{
		PoolLock l;

		Pool& pool = instance();

		pool.allocated++;

		if (!pool.store.empty())
		{
			T* result = pool.store.back();
			pool.store.pop_back();
			return result;
		}

		T* result = BaseNew<T,true>::internal_allocate();
		PoolPolicy<T>::setPooled(result, true);
		return result;
	}
	else
	{
		return BaseNew<T,true>::internal_allocate();
	}
}

template<class T>
void Pool<T>::release(T* p)
{
	if (PoolPolicy<T>::isPooled(p))
	{
		PoolPolicy<T>::recycle(p);

		PoolLock l;

		Pool& pool = instance();
		pool.store.push_back(p);
		pool.allocated--;
	}
	else
	{
		BaseNew<T,true>::internal_release(p);
	}
}

template<class T>
Pool<T>& Pool<T>::instance()
{
	static Pool thePool;
	return thePool;
}

template<class T>
Pool<T>::Pool()
	: allocated(0)
{
	store.reserve(PoolPolicy<T>::ReserveCount);
}

template<class T>
Pool<T>::~Pool() 
{
	for (PoolVector::iterator it = store.begin(), end = store.end(); it != end; ++it)
		BaseNew<T,true>::internal_release(*it);
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif

