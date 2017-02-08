
#include <cstddef>

// a compile-time index sequence similar to c++14 std::index_sequence

template<std::size_t ... n>
struct seq {

    template<std::size_t x>
    using push = seq<n..., x>;
};

template<std::size_t i>
struct gen_seq {
    using type = typename gen_seq<i-1>::type::template push<i>;
};


template<>
struct gen_seq<0> {
    using type = seq<0>;
};

template<class ...T>
static typename gen_seq<sizeof...(T) - 1>::type make_sequence() { return {}; }

