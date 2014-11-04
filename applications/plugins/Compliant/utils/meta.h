#ifndef COMPLIANT_UTILS_META_H
#define COMPLIANT_UTILS_META_H

#include <sofa/core/ObjectFactory.h>

// meta programming for easier template instanciations
namespace meta {
  
  // cons, empty list is void
  template<class Head, class Tail = void>
  struct cons;


  // list concatenation
  template<class LHS = void, class RHS = void> struct concat;

  template<>
  struct concat<> {
	typedef void type;
  };

  template<class LHead, class LTail, class RHS>
  struct concat< cons<LHead, LTail>,
				 RHS > {
	typedef cons<LHead, typename concat<LTail, RHS>::type > type;
  };

  template<class RHS>
  struct concat<void, RHS> {
	typedef RHS type;
  };


  // list map
  template< template<class> class T, class = void>
  struct map;

  template< template<class> class T>
  struct map<T> {
	typedef void type;
  };

  template< template<class> class T, class Head, class Tail>
  struct map<T, cons<Head, Tail> > {
	typedef cons< typename T<Head>::type,
				  typename map<T, Tail>::type > type;
  };


  // build a cons list from user list type
  template<class> struct make;

  // tag for list type
  struct list;

  template<>
  struct make< list() > {
	typedef void type;
  };

  template<class T>
  struct make< list( T ) > {
	typedef cons<T> type;
  };

  template<class T1, class T2>
  struct make< list( T1, T2 ) > {
	typedef cons<T1, typename make< list(T2) >::type > type;
  };

  template<class T1, class T2, class T3>
  struct make< list( T1, T2, T3 ) > {
	typedef cons<T1, typename make< list(T2, T3) >::type > type;
  };

  template<class T1, class T2, class T3, class T4>
  struct make< list( T1, T2, T3, T4 ) > {
	typedef cons<T1, typename make< list(T2, T3, T4) >::type > type;
  };

  template<class T1, class T2, class T3, class T4, class T5>
  struct make< list( T1, T2, T3, T4, T5 ) > {
	typedef cons<T1, typename make< list(T2, T3, T4, T5) >::type > type;
  };

  template<class T1, class T2, class T3, class T4, class T5, class T6>
  struct make< list( T1, T2, T3, T4, T5, T6 ) > {
	typedef cons<T1, typename make< list(T2, T3, T4, T5, T6) >::type > type;
  };

  template<class T1, class T2, class T3, class T4, class T5, class T6, class T7>
  struct make< list( T1, T2, T3, T4, T5, T6, T7 ) > {
	typedef cons<T1, typename make< list(T2, T3, T4, T5, T6, T7) >::type > type;
  };

  template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
  struct make< list( T1, T2, T3, T4, T5, T6, T7, T8 ) > {
	typedef cons<T1, typename make< list(T2, T3, T4, T5, T6, T7, T8) >::type > type;
  };


  // if you need more, feel free :-)


  // an integer wrapped in a type
  template<int> struct Int;


  // a pair for template instanciation
  template<class, class> struct pair;


  // make all possible pairs from two type lists
  template<class LHS, class RHS = void>
  struct make_pairs;


  template<class T>
  struct make_pairs< cons<T> > {
	typedef void type;
  };


  template<class LHSHead, class LHSTail,
		   class RHS>
  struct make_pairs< cons<LHSHead, LHSTail>,
					 RHS> {

	typedef typename concat< typename make_pairs< cons<LHSHead>, RHS >::type,
							 typename make_pairs< LHSTail, RHS >::type >::type type;

  };

  template<class T, class RHSHead, class RHSTail>
  struct make_pairs< cons<T>,
					 cons< RHSHead, RHSTail > > {

	typedef cons< pair<T, RHSHead>,
				  typename make_pairs<cons<T>, RHSTail>::type > type;
  
  };


  // all possible pairs in a list (i.e. for collision detection)
  template<class List = void> struct make_combinations;

  template<> struct make_combinations<> {
	typedef void type;
  };
  
  template<class Head, class Tail>
  struct make_combinations< cons<Head, Tail> > {

	typedef typename concat< typename make_pairs< cons<Head>, cons<Head, Tail> >::type,
							 typename make_combinations< Tail >::type >::type type;

  };


  // explicit instantiations of this class will trigger explicit
  // instantiations of (not already explicitely instantiated) members,
  // as mandated by the standard.
  template<class List = void>
  struct instantiate {
	
	template<class T>
	instantiate(T& arg) { }

	template<class T>
	instantiate(const T& arg) { }

	
	template<class T1, class T2>
	instantiate(T1& arg1, T2& arg2) { }

	template<class T1, class T2>
	instantiate(T1& arg1, const T2& arg2) { }

  };

  template<class Head, class Tail>
  struct instantiate< cons<Head, Tail> >  {

	Head head;
	instantiate<Tail> tail;

	template<class T>
	instantiate(T& arg) : head(arg), tail(arg) { }

	template<class T>
	instantiate(const T& arg) : head(arg), tail(arg) { }


	template<class T1, class T2>
	instantiate(T1& arg1, T2& arg2) : head(arg1, arg2),
									  tail(arg1, arg2) { }

	template<class T1, class T2>
	instantiate(T1& arg1, const T2& arg2) : head(arg1, arg2),
											tail(arg1, arg2) { }

	// TODO add more if needed
  };

  

  namespace impl {

#ifdef SOFA_FLOAT
#define __SOFA_FLOAT 1
#else
#define __SOFA_FLOAT 0
#endif

#ifdef SOFA_DOUBLE
#define __SOFA_DOUBLE 1
#else
#define __SOFA_DOUBLE 0
#endif


  
	template<int SofaFloat = __SOFA_FLOAT, int SofaDouble = __SOFA_DOUBLE>
	struct sofa_real_list;

	template<>
	struct sofa_real_list<0, 0> {
	  typedef make< list(double, float) >::type type;
	};

	template<>
	struct sofa_real_list<1, 0> {
	  typedef make< list(float) >::type type;
	};

	template<>
	struct sofa_real_list<0, 1> {
	  typedef make< list(double) >::type type;
	};
  
  }

  // a type list of real types 
  typedef impl::sofa_real_list<>::type sofa_real_list;


  // add type T to a register object
  template<class T>
  struct register_object {

	typedef register_object type;
	register_object(sofa::core::RegisterObject& object ) {
	  object.add<T>();
	}
  
  };


  // add every type in a list to a register object
  template<class List>
  struct register_list {

	sofa::core::RegisterObject object;
	instantiate< typename map<register_object, List>::type > apply;

	register_list(const std::string& description):
	  object( description ),
	  apply( object ) {

	}

  };


  // some helpers
  template<class C, int I>
  struct make_vectypes {
	typedef sofa::defaulttype::Vec<I, C> vec_type;
	typedef sofa::defaulttype::StdVectorTypes<vec_type, vec_type, C> type;
  };


}

#endif
