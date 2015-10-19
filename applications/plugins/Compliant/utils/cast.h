#ifndef UTILS_CAST_H
#define UTILS_CAST_H

#include <cassert>

// dynamic cast + assert 
template<class T, class U>
T* safe_cast(U* what) {
	T* res = dynamic_cast<T*>(what);
	assert( res );
	return res;
}


// static_cast + assert 
template<class T, class U>
T* down_cast(U* what) {
    assert( dynamic_cast<T*>(what) );
    return static_cast<T*>(what);
}


#endif
