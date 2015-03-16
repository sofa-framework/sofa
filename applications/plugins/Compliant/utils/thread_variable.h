#ifndef COMPLIANT_THREAD_VARIABLE_H
#define COMPLIANT_THREAD_VARIABLE_H

#if defined(WIN32)
#include <omp.h>
#endif

template<class A>
class thread_variable {
	typedef std::map<int, A*> value_type;
	value_type value;
	
	static int id() { 

#ifdef _OPENMP
		int res = omp_get_thread_num();
#else
		int res = 0;
#endif
		
		return res;
	}

	A* get() {
		int i = id();

		typename value_type::iterator it;

		it = value.find( i );
		
		if( it == value.end() ) {
			A* a = new A;
			value[ i ] = a;
			return a;
		} else {
			return it->second;
		}
	}

public:
	
	A* operator->() { 
		A* res = 0;
#ifdef _OPENMP
#pragma omp critical
#endif
		res = get(); 
		
		return res;
	}
	
	void clear() {  
		for(typename value_type::iterator it = value.begin(), end = value.end(); it != end; ++it) {
			delete it->second;
		}
		value.clear();
	}

	~thread_variable() { clear(); }
};


#endif
