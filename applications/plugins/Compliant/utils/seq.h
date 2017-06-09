
#include <cstddef>

// a compile-time index sequence similar to c++14 std::index_sequence

template<std::size_t ... N>
struct seq { };

template<class s1, class s2>
struct merge_sequences_type;

template<std::size_t ... I1, std::size_t ... I2>
struct merge_sequences_type< seq<I1...>, seq<I2...> >{
    using type = seq<I1..., I2...>;
};

template<std::size_t S, std::size_t E>
struct make_sequence_type {

    static constexpr std::size_t pivot = E/2;
    using lhs_type = typename make_sequence_type<S, pivot>::type;
    using rhs_type = typename make_sequence_type<pivot + 1, E>::type;    
    
    using type = typename merge_sequences_type< lhs_type, rhs_type>::type;
};

template<std::size_t I>
struct make_sequence_type<I, I> {
    using type = seq<I>;
};

template<class ...T>
static typename make_sequence_type<0, sizeof...(T)>::type make_sequence() { return {}; }


template< template<std::size_t ... I> class cls, class seq >
struct instantiate_sequence_type;


template< template<std::size_t ... I> class cls, std::size_t ... I>
struct instantiate_sequence_type<cls, seq<I...> > {
    using type = cls<I...>;
};


