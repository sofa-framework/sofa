#include <cstddef>


template<class T>
class view {
    T* m_data;
    std::size_t m_size;
public:

    view(T* data, std::size_t size)
        : m_data(data),
          m_size(size) {

    }
    
    T* data() { return m_data; }

    std::size_t size() const { return m_size; }
    
    T* begin() { return data(); }
    T* end() { return data() + size(); }
    T& operator[](std::size_t i) {
        assert(i < size());
        return m_data[i];
    }

    // TODO: only if T is not already const?
    const T& operator[](std::size_t i) const {
        assert(i < size());
        return m_data[i];
    }

    const T* begin() const { return data(); }
    const T* end() const { return data() + size(); }
    const T* data() const{ return m_data; }    
};
    

namespace detail {

template<class U>
struct forward_const {
    using coord_type = typename U::Coord;
    using deriv_type = typename U::Deriv;    
};

template<class U>
struct forward_const<const U> {
    using coord_type = const typename U::Coord;
    using deriv_type = const typename U::Deriv;    
};


};


template<class U>
using coord_view = view< typename detail::forward_const<U>::coord_type >;

template<class U>
using deriv_view = view< typename detail::forward_const<U>::deriv_type >;

