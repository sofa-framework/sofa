
#ifndef __ITERATORIZE__
#define __ITERATORIZE__
/**
 * template classs that add iterator to Traversor
 * to allow the use of c++11 syntax for (auto d : v)
 */
template <typename TRAV>
class Iteratorize: public TRAV
{
public:
	typedef typename TRAV::MapType MAP;
	typedef typename TRAV::IterType ITER;
	typedef typename TRAV::ParamType PARAM;

	Iteratorize(const MAP& map, PARAM p):
		TRAV(map,p),m_begin(this,TRAV::begin()),m_end(this,TRAV::end())
	{}


	class iterator
	{
		Iteratorize<TRAV>* m_ptr;
		ITER m_index;

	public:

		inline iterator(Iteratorize<TRAV>* p, ITER i): m_ptr(p),m_index(i){}

		inline iterator& operator++()
		{
			m_index = m_ptr->next();
			return *this;
		}

		inline ITER& operator*()
		{
			return m_index;
		}

		inline bool operator!=(const iterator& it)
		{
			return m_index.dart != it.m_index.dart;
		}

	};

	inline iterator begin()
	{
		return m_begin;
	}

	inline iterator end()
	{
		return m_end;
	}

protected:
	iterator m_begin;
	iterator m_end;
};

#endif
