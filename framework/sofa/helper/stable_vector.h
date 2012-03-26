/* Stable vector, using either the version from boost 1.48+, or the original code below.
 *
 * Copyright 2008 Joaquín M López Muñoz.
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef SOFA_HELPER_STABLE_VECTOR_H
#define SOFA_HELPER_STABLE_VECTOR_H

#include <boost/version.hpp>

#if BOOST_VERSION >= 104800

// Use boost version

#include <boost/container/stable_vector.hpp>

namespace sofa
{
namespace helper
{

template<class T, class A = std::allocator<T> >
class stable_vector : public boost::container::stable_vector<T,A>
{
public:
    typedef boost::container::stable_vector<T,A> Inherit;
    typedef typename Inherit::size_type size_type;
    stable_vector() {}
    explicit stable_vector(size_type n) : Inherit(n) {}
    stable_vector(size_type n, const T& t) : Inherit(n,t) {}
    template<class InputIterator>
    stable_vector(InputIterator first, InputIterator last) : Inherit(first,last) {}
    stable_vector(const stable_vector& x) : Inherit(x) {}
};

} // namespace helper
} // namespace sofa

#else // if BOOST_VERSION >= 104800

#include <algorithm>
#include <stdexcept>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/not.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <vector>

#if defined(STABLE_VECTOR_ENABLE_INVARIANT_CHECKING)
#include <boost/assert.hpp>
#endif

namespace sofa
{
namespace helper
{

namespace stable_vector_detail
{

template<typename T,typename Value>
class iterator;

template<typename T>
struct node_type
{
    void**                           up;
    typename boost::aligned_storage<
    sizeof(T),
           boost::alignment_of<T>::value
           >::type                          spc;

    T& value() {return *static_cast<T*>(static_cast<void*>(&spc));}
};

class node_access
{
public:
    template<typename T,typename Value>
    static typename iterator<T,Value>::node_type* get(
        const iterator<T,Value>& it)
    {
        return it.pn;
    }
};

template<typename T,typename Value>
class iterator:
    public boost::iterator_facade<
    iterator<T,Value>,Value,std::random_access_iterator_tag>
{
    typedef node_type<T> node_type;

public:
    iterator() {}
    explicit iterator(node_type* pn):pn(pn) {}
    iterator(const iterator<T,T>& x):pn(node_access::get(x)) {}

private:
    static node_type* node_ptr(void* p) {return static_cast<node_type*>(p);}

    friend class boost::iterator_core_access;

    Value& dereference()const {return pn->value();}
    bool equal(const iterator& x)const {return pn==x.pn;}
    void increment() {pn=node_ptr(*(pn->up+1));}
    void decrement() {pn=node_ptr(*(pn->up-1));}
    void advance(std::ptrdiff_t n) {pn=node_ptr(*(pn->up+n));}
    std::ptrdiff_t distance_to(const iterator& x)const {return x.pn->up-pn->up;}

    friend class node_access;

    node_type* pn;
};

} //namespace stable_vector_detail

#if defined(STABLE_VECTOR_ENABLE_INVARIANT_CHECKING)
#define STABLE_VECTOR_CHECK_INVARIANT \
invariant_checker BOOST_JOIN(check_invariant_,__LINE__)(*this); \
BOOST_JOIN(check_invariant_,__LINE__).touch();
#else
#define STABLE_VECTOR_CHECK_INVARIANT
#endif

template<typename T,typename Allocator=std::allocator<T> >
class stable_vector
{
    typedef stable_vector_detail::node_type<T>        node_type;
    typedef std::vector<
    void*,
    typename Allocator::
    template rebind<void*>::other
    >                                                 impl_type;
    typedef typename impl_type::iterator              impl_iterator;
    typedef typename impl_type::const_iterator        const_impl_iterator;

public:
    // types:

    typedef typename Allocator::reference             reference;
    typedef typename Allocator::const_reference       const_reference;
    typedef stable_vector_detail::iterator<T,T>       iterator;
    typedef stable_vector_detail::iterator<T,const T> const_iterator;
    typedef typename impl_type::size_type             size_type;
    typedef typename iterator::difference_type        difference_type;
    typedef T                                         value_type;
    typedef Allocator                                 allocator_type;
    typedef typename Allocator::pointer               pointer;
    typedef typename Allocator::const_pointer         const_pointer;
    typedef std::reverse_iterator<iterator>           reverse_iterator;
    typedef std::reverse_iterator<const_iterator>     const_reverse_iterator;

    // construct/copy/destroy:

    explicit stable_vector(const Allocator& al=Allocator()):
        al(al),impl((size_type)1,0,al)
    {
        create_end_node();
        STABLE_VECTOR_CHECK_INVARIANT;
    }

    stable_vector(size_type n,const T& t=T(),const Allocator& al=Allocator()):
        al(al),impl(al)
    {
        range_ctor_not_iter(n,t);
        STABLE_VECTOR_CHECK_INVARIANT;
    }

    template <class InputIterator>
    stable_vector(
        InputIterator first,InputIterator last,const Allocator& al=Allocator()):
        al(al),impl(al)
    {
        range_ctor_iter(
            first,last,boost::mpl::not_<boost::is_integral<InputIterator> >());
        STABLE_VECTOR_CHECK_INVARIANT;
    }

    stable_vector(const stable_vector<T,Allocator>& x):
        al(x.al),impl(x.impl.size(),0,al)
    {
        size_type i=0,n=impl.size()-1;
        try
        {
            while(i<n)
            {
                impl[i]=new_node(&impl[i],x[i]);
                ++i;
            }
            create_end_node();
        }
        catch(...)
        {
            while(i--)delete_node(impl[i]);
            throw;
        }
        STABLE_VECTOR_CHECK_INVARIANT;
    }

    ~stable_vector()
    {
        clear();
        destroy_end_node();
    }

    stable_vector& operator=(stable_vector x)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        swap(x);
        return *this;
    }

    template<typename InputIterator>
    void assign(InputIterator first,InputIterator last)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        erase(begin(),end());
        insert(begin(),first,last);
    }

    void assign(size_type n,const T& t)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        erase(begin(),end());
        insert(begin(),n,t);
    }

    allocator_type get_allocator()const {return al;}

    // iterators:

    iterator        begin() {return iterator(node_ptr(impl.front()));}
    const_iterator  begin()const {return const_iterator(node_ptr(impl.front()));}
    iterator        end() {return iterator(node_ptr(impl.back()));}
    const_iterator  end()const {return const_iterator(node_ptr(impl.back()));}

    reverse_iterator       rbegin() {return reverse_iterator(end());}
    const_reverse_iterator rbegin()const {return const_reverse_iterator(end());}
    reverse_iterator       rend() {return reverse_iterator(begin());}
    const_reverse_iterator rend()const {return const_reverse_iterator(begin());}

    const_iterator         cbegin()const {return begin();}
    const_iterator         cend()const {return end();}
    const_reverse_iterator crbegin()const {return rbegin();}
    const_reverse_iterator crend()const {return rend();}

    // capacity:

    size_type size()const {return impl.size()-1;}
    size_type max_size()const {return impl.max_size()-1;}

    void resize(size_type n,const T& t=T())
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        if(n>size())insert(end(),n-size(),t);
        else if(n<size())erase(begin()+n,end());
    }

    size_type capacity()const {return impl.capacity()-1;}
    bool empty()const {return impl.size()==1;}

    void reserve(size_type n)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        if(n>capacity())
        {
            impl.reserve(n+1);
            align_nodes(impl.begin(),impl.end());
        }
    }

    // element access:

    reference operator[](size_type n) {return value(impl[n]);}
    const_reference operator[](size_type n)const {return value(impl[n]);}

    const_reference at(size_type n)const
    {
        if(n>=size())
            throw std::out_of_range("invalid subscript at stable_vector::at");
        return operator[](n);
    }

    reference at(size_type n)
    {
        if(n>=size())
            throw std::out_of_range("invalid subscript at stable_vector::at");
        return operator[](n);
    }

    reference front() {return value(impl.front());}
    const_reference front()const {return value(impl.front());}
    reference back() {return value(*(&impl.back()-1));}
    const_reference back()const {return value(*(&impl.back()-1));}

    // modifiers:

    void push_back(const T& t) {insert(end(),t);}
    void pop_back() {erase(end()-1);}

    iterator insert(const_iterator position,const T& t)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        difference_type d=position-begin();
        impl_iterator   it;
        if(impl.capacity()>impl.size())
        {
            it=impl.insert(impl.begin()+d,0);
            try
            {
                *it=new_node(&*it,t);
            }
            catch(...)
            {
                impl.erase(it);
                throw;
            }
            align_nodes(it+1,impl.end());
        }
        else
        {
            it=impl.insert(impl.begin()+d,0);
            try
            {
                *it=new_node(0,t);
            }
            catch(...)
            {
                impl.erase(it);
                align_nodes(impl.begin(),impl.end());
                throw;
            }
            align_nodes(impl.begin(),impl.end());
        }
        return iterator(node_ptr(*it));
    }

    void insert(const_iterator position,size_type n,const T& t)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        insert_not_iter(position,n,t);
    }

    template <class InputIterator>
    void insert(const_iterator position,InputIterator first,InputIterator last)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        insert_iter(
            position,first,last,
            boost::mpl::not_<boost::is_integral<InputIterator> >());
    }

    iterator erase(const_iterator position)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        difference_type d=position-begin();
        impl_iterator   it=impl.begin()+d;
        delete_node(*it);
        impl.erase(it);
        align_nodes(impl.begin()+d,impl.end());
        return begin()+d;
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        difference_type d1=first-begin(),d2=last-begin();
        impl_iterator   it1=impl.begin()+d1,it2=impl.begin()+d2;
        for(impl_iterator it=it1; it!=it2; ++it)delete_node(*it);
        impl.erase(it1,it2);
        align_nodes(impl.begin()+d1,impl.end());
        return begin()+d1;
    }

    void swap(stable_vector& x)
    {
        STABLE_VECTOR_CHECK_INVARIANT;
        swap_impl(*this,x);
    }

    void clear() {erase(begin(),end());}

private:
    static node_type* node_ptr(void* p)
    {
        return static_cast<node_type*>(p);
    }

    static value_type& value(void* p)
    {
        return node_ptr(p)->value();
    }

    void create_end_node()
    {
        node_type* p=al.allocate(1);
        impl.back()=p;
        p->up=&impl.back();
    }

    void destroy_end_node()
    {
        al.deallocate(node_ptr(impl.back()),1);
    }

    void* new_node(void** up,const T& t)
    {
        node_type* p=al.allocate(1);
        try
        {
            p->up=up;
            allocator_type(al).construct(&p->value(),t);
        }
        catch(...)
        {
            al.deallocate(p,1);
            throw;
        }
        return p;
    }

    void delete_node(void* p)
    {
        allocator_type(al).destroy(&value(p));
        al.deallocate(node_ptr(p),1);
    }

    static void align_nodes(impl_iterator first,impl_iterator last)
    {
        while(first!=last)
        {
            node_ptr(*first)->up=&*first;
            ++first;
        }
    }

    void range_ctor_not_iter(size_type n,const T& t)
    {
        impl.assign(n+1,0);
        size_type i=0;
        try
        {
            while(i<n)
            {
                impl[i]=new_node(&impl[i],t);
                ++i;
            }
            create_end_node();
        }
        catch(...)
        {
            while(i--)delete_node(impl[i]);
            throw;
        }
    }

    template <class InputIterator>
    void range_ctor_iter(
        InputIterator first,InputIterator last,boost::mpl::true_)
    {
        typedef typename std::iterator_traits<
        InputIterator>::iterator_category    category;
        range_ctor_iter(first,last,category());
    }

    template <class InputIterator>
    void range_ctor_iter(
        InputIterator first,InputIterator last,std::input_iterator_tag)
    {
        size_type i=0;
        try
        {
            while(first!=last)
            {
                impl.push_back(0);
                impl.back()=new_node(0,*first++);
                ++i;
            }
            impl.push_back(0);
            create_end_node();
        }
        catch(...)
        {
            while(i--)delete_node(impl[i]);
            throw;
        }
        align_nodes(impl.begin(),impl.end());
    }

    template <class InputIterator>
    void range_ctor_iter(
        InputIterator first,InputIterator last,std::forward_iterator_tag)
    {
        size_type n=(size_type)std::distance(first,last);
        impl.assign(n+1,0);
        size_type i=0;
        try
        {
            while(first!=last)
            {
                impl[i]=new_node(&impl[i],*first++);
                ++i;
            }
            create_end_node();
        }
        catch(...)
        {
            while(i--)delete_node(impl[i]);
            throw;
        }
    }

    template <class InputIterator>
    void range_ctor_iter(
        InputIterator first,InputIterator last,boost::mpl::false_)
    {
        range_ctor_not_iter(first,last);
    }

    void insert_not_iter(const_iterator position,size_type n,const T& t)
    {
        difference_type d=position-begin();
        if(impl.capacity()>=impl.size()+n)
        {
            impl.insert(impl.begin()+d,n,0);
            impl_iterator it=impl.begin()+d;
            size_type i=0;
            try
            {
                while(i<n)
                {
                    *(it+i)=new_node(&*(it+i),t);
                    ++i;
                }
            }
            catch(...)
            {
                impl.erase(it+i,it+n);
                align_nodes(it+i,impl.end());
                throw;
            }
            align_nodes(it+n,impl.end());
        }
        else
        {
            impl.insert(impl.begin()+d,n,0);
            impl_iterator it=impl.begin()+d;
            size_type i=0;
            try
            {
                while(i<n)
                {
                    *(it+i)=new_node(&*(it+i),t);
                    ++i;
                }
            }
            catch(...)
            {
                impl.erase(it+i,it+n);
                align_nodes(impl.begin(),it);
                align_nodes(it+i,impl.end());
                throw;
            }
            align_nodes(impl.begin(),it);
            align_nodes(it+n,impl.end());
        }
    }

    template <class InputIterator>
    void insert_iter(
        const_iterator position,InputIterator first,InputIterator last,
        boost::mpl::true_)
    {
        typedef typename std::iterator_traits<
        InputIterator>::iterator_category    category;
        insert_iter(position,first,last,category());
    }

    template <class InputIterator>
    void insert_iter(
        const_iterator position,InputIterator first,InputIterator last,
        std::input_iterator_tag)
    {
        difference_type d=position-begin();
        size_type       c=impl.capacity();
        size_type       i=0;
        try
        {
            while(first!=last)
            {
                impl_iterator it=impl.insert(impl.begin()+d+i,0);
                try
                {
                    *it=new_node(&*it,*first++);
                }
                catch(...)
                {
                    impl.erase(it);
                    throw;
                }
                ++i;
            }
        }
        catch(...)
        {
            if(c==impl.capacity())
            {
                align_nodes(impl.begin()+d+i,impl.end());
            }
            else
            {
                align_nodes(impl.begin(),impl.end());
            }
            throw;
        }
        if(c==impl.capacity())
        {
            align_nodes(impl.begin()+d+i,impl.end());
        }
        else
        {
            align_nodes(impl.begin(),impl.end());
        }
    }

    template <class InputIterator>
    void insert_iter(
        const_iterator position,InputIterator first,InputIterator last,
        std::forward_iterator_tag)
    {
        size_type       n=(size_type)std::distance(first,last);
        difference_type d=position-begin();
        if(impl.capacity()>=impl.size()+n)
        {
            impl.insert(impl.begin()+d,n,0);
            impl_iterator it=impl.begin()+d;
            size_type i=0;
            try
            {
                while(first!=last)
                {
                    *(it+i)=new_node(&*(it+i),*first++);
                    ++i;
                }
            }
            catch(...)
            {
                impl.erase(it+i,it+n);
                align_nodes(it+i,impl.end());
                throw;
            }
            align_nodes(it+n,impl.end());
        }
        else
        {
            impl.insert(impl.begin()+d,n,0);
            impl_iterator it=impl.begin()+d;
            size_type i=0;
            try
            {
                while(first!=last)
                {
                    *(it+i)=new_node(&*(it+i),*first++);
                    ++i;
                }
            }
            catch(...)
            {
                impl.erase(it+i,it+n);
                align_nodes(impl.begin(),it);
                align_nodes(it+i,impl.end());
                throw;
            }
            align_nodes(impl.begin(),it);
            align_nodes(it+n,impl.end());
        }
    }

    template <class InputIterator>
    void insert_iter(
        const_iterator position,InputIterator first,InputIterator last,
        boost::mpl::false_)
    {
        insert_not_iter(position,first,last);
    }

    static void swap_impl(stable_vector& x,stable_vector& y)
    {
        using std::swap;
        swap(x.al,y.al);
        swap(x.impl,y.impl);
    }

#if defined(STABLE_VECTOR_ENABLE_INVARIANT_CHECKING)
    bool invariant()const
    {
        if(impl.size()<1)return false;
        for(const_impl_iterator it=impl.begin(),it_end=impl.end();
            it!=it_end; ++it)
        {
            if(node_ptr(*it)->up!=&*it)return false;
        }
        return true;
    }

    class invariant_checker:private boost::noncopyable
    {
        const stable_vector* p;
    public:
        invariant_checker(const stable_vector& v):p(&v) {}
        ~invariant_checker() {BOOST_ASSERT(p->invariant());}
        void touch() {}
    };
#endif

    typename allocator_type::
    template rebind<node_type>::other al;
    impl_type                           impl;
};

template <typename T,typename Allocator>
bool operator==(
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return x.size()==y.size()&&std::equal(x.begin(),x.end(),y.begin());
}

template <typename T,typename Allocator>
bool operator< (
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end());
}

template <typename T,typename Allocator>
bool operator!=(
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return !(x==y);
}

template <typename T,typename Allocator>
bool operator> (
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return y<x;
}

template <typename T,typename Allocator>
bool operator>=(
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return !(x<y);
}

template <typename T,typename Allocator>
bool operator<=(
    const stable_vector<T,Allocator>& x,const stable_vector<T,Allocator>& y)
{
    return !(x>y);
}

// specialized algorithms:

template <typename T, typename Allocator>
void swap(stable_vector<T,Allocator>& x,stable_vector<T,Allocator>& y)
{
    x.swap(y);
}

#undef STABLE_VECTOR_CHECK_INVARIANT

} // namespace helper
} // namespace sofa

#endif // if BOOST_VERSION >= 104800

#endif
