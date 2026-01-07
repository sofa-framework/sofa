#pragma once

#include <iterator>

namespace sofa::helper
{

/**
 * Simple alternative to std::ranges::iota_view, compatible with pre-C++20
 */
template <typename T>
class IotaView
{
public:
    struct iota_iterator
    {
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using reference = T&;
        using pointer = T*;

        iota_iterator() = default;
        explicit iota_iterator(T val) : val_(val) {}

        T operator*() const { return val_; }
        iota_iterator& operator++() { ++val_; return *this; }
        bool operator!=(const iota_iterator& other) const { return val_ != other.val_; }

    private:
        T val_{};
    };

    using iterator = iota_iterator;
    using value_type = T;

    IotaView(T start, T end) : m_start(start), m_end(end) {}

    iterator begin() const { return iterator(m_start); }
    iterator end() const { return iterator(m_end); }

    [[nodiscard]] T size() const { return m_end - m_start; }
    [[nodiscard]] T operator[](T i) const { return m_start + i; }
    T front() const { return m_start; }
    T back() const { return m_end - 1; }
    [[nodiscard]] bool empty() const { return m_start == m_end; }

private:
    T m_start;
    T m_end;
};


}
