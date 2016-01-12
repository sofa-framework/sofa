
#ifndef SOFA_NOPREALLOCATIONCONTAINERS_H
#define SOFA_NOPREALLOCATIONCONTAINERS_H



#include "ImageTypes.h"
#include <vector>

namespace sofa
{

namespace helper
{



/// minimalist vector with a tight memory array
/// the particularity is not to pre-allocate (no unused memory is allocated without a user choice)
/// not efficient to add an entry (since it must resize the array)
/// @todo: move it somewhere in sofa::helper
template<class T>
class NoPreallocationVector
{

public:

    /// default constructor, no allocation, no initialization
    NoPreallocationVector() : _array(0), _size(0) {}
    /// given allocation, no initialization
    NoPreallocationVector( size_t size )
    {
        _size = size;
        if( !size ) { _array=0; return; }
        _array = new T[_size];
    }
    /// default destructor
    ~NoPreallocationVector() { if( _array ) delete [] _array; }

    /// copy constructor
    NoPreallocationVector( const NoPreallocationVector<T>& c )
    {
        _array = new T[c._size];
        _size = c._size;
        memcpy( _array, c._array, _size*sizeof(T) );
    }

    /// free memory
    void clear()
    {
        if( _array )
        {
            delete [] _array;
            _array = 0;
            _size = 0;
        }
    }

    /// clone
    void operator=( const NoPreallocationVector<T>& c )
    {
        if( _array ) delete [] _array;
        if( c.empty() ) { _array=0; _size=0; return; }
        _array = new T[c._size];
        _size = c._size;
        memcpy( _array, c._array, _size*sizeof(T) );
    }

    /// comparison
    bool operator==( const NoPreallocationVector<T>& c ) const
    {
        if( _size != c._size ) return false;
        for( unsigned i=0 ; i<_size ; ++i )
            if( _array[i] != c._array[i] ) return false;
        return true;
    }

    /// difference
    bool operator!=( const NoPreallocationVector<T>& c ) const
    {
        return !(*this==c);
    }

    /// add a entry at the end
    /// @warning this has bad performances, since resizing/reallocating the array is necessary
    void push_back( const T& v )
    {
        if( !_array ) // alloc
        {
            _array = new T[_size];
        }
        else // realloc
        {
            T* tmp = new T[_size+1];
            memcpy( tmp, _array, _size*sizeof(T) );
            delete [] _array;
            _array = tmp;
        }
        // push back the new element
        _array[_size++] = v;
    }

    /// @warning data is lost
    void resize( size_t newSize )
    {
        _size = newSize;
        if( _array ) delete [] _array;
        if( !newSize ) { _array=0; return; }
        _array = new T[_size];
    }

    /// already-existing data is preserved
    void resizeAndKeep( size_t newSize )
    {
        if( !newSize )
        {
            delete [] _array;
            _array = 0;
            return;
        }

        if( !_array )
        {
            _size = newSize;
            _array = new T[_size];
        }
        else
        {
            T* tmpArray = new T[newSize];
            memcpy( tmpArray, _array, std::min(newSize,_size)*sizeof(T) );
            _size = newSize;
            delete [] _array;
            _array = tmpArray;
        }
    }

    /// \return the index of the first occurence, if !present \return -1
    int find( const T& v ) const
    {
        for( unsigned i = 0 ; i<_size ; ++i )
            if( _array[i]==v ) return (int)i;
        return -1;
    }

    /// \return true iff v is present
    bool isPresent( const T& v ) const
    {
        for( unsigned i = 0 ; i<_size ; ++i )
            if( _array[i]==v ) return true;
        return false;
    }

    /// \return the index of the occurence, if !present \return -1
    int getOffset( const T* v ) const
    {
        int offset = v - _array;
        if( offset>=0 && (unsigned)offset<_size ) return offset;
        return -1;
    }

    /// entry accessor
    T& operator[]( size_t index ) { /*assert( index < _size );*/ return _array[ index ]; }
    /// entry const accessor
    const T& operator[]( size_t index ) const { /*assert( index < _size );*/ return _array[ index ]; }

    /// first entry accessor
    T& first() { assert(_size>0); return _array[0]; }
    /// first entry const accessor
    const T& first() const { assert(_size>0); return _array[0]; }
    /// last entry accessor
    T& last() { assert(_size>0); return _array[_size-1]; }
    /// last entry const accessor
    const T& last() const { assert(_size>0); return _array[_size-1]; }

    /// @returns the entry number (effective and allocated)
    inline size_t size() const { return _size; }
    /// @returns true iff the vector is empty (no entry)
    inline bool empty() const { return !_size; }

    /// fill all array entries with the given value v
    void fill( const T&v )
    {
        for( unsigned i=0 ; i<_size ; ++i )
            _array[i] = v;
    }

    /// read @warning does nothing for now, just needed to put a NoPreallocationVector in a Data
    friend std::istream& operator >> ( std::istream& in, NoPreallocationVector<T>& /*c*/ )
    {
//        T t = T();
//        c.clear();
//        while( in >> t ) c.push_back( t );
//        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

    /// write @warning does nothing for now, just needed to put a NoPreallocationVector in a Data
    friend std::ostream& operator << ( std::ostream& out, const NoPreallocationVector<T>& /*c*/ )
    {
//        for( unsigned i=0 ; i<c._size-1 ; ++i )
//        {
//            out << c._array[i] << " ";
//        }
//        out << c._array[c._size-1];
        return out;
    }

protected:

    T* _array; ///< the array where to store the entries
    size_t _size; ///< the array size

}; // class NoPreallocationVector





/// a key -> several elements
/// the Key type must have the operator<
template <class _Key, class _T>
class MultiMap_sortedList
{

public :

    typedef _T T; ///< the value type
    typedef _Key Key; ///< the key type

    struct Element
    {
        Element() : next(0) {}
        Element( const Key& k, Element* n = 0 ) : key(k), next(n) {}
        Element( const Key& k, const T& v, Element* n = 0 ) : key(k), value(v), next(n) {}
        Key key; ///< the element key
        T value; ///< the element value
        Element* next; ///< the next list element
    };


    MultiMap_sortedList() : _first(0) {}

    MultiMap_sortedList( const MultiMap_sortedList<Key,T>& other )
    {
        if( !other._first ) { _first=0; return; }

        _first = new Element( other._first->key, other._first->value );

        Element* otherNext = other._first->next;
        Element* cur = _first;

        while( otherNext )
        {
            cur->next = new Element( otherNext->key, otherNext->value );
        }
    }



    ~MultiMap_sortedList() { clear(); }


    void clear()
    {
        if( _first )
        {
            Element* cur = _first->next;
            Element* prev = _first;

            while( cur )
            {
                Element* next = cur->next;
                delete cur;
                cur = prev->next = next;
            }

            delete _first;
            _first = 0;
        }
    }



    /// @returns a vector of const list elements with the given key
    void find( const Key& k, std::vector<const Element*>& result ) const
    {
        Element* e = _first;
        while( e )
        {
            if( e->key>k ) break;
            if( e->key==k ) { result.push_back(e); }
            e = e->next;
        }
    }

    /// @returns a vector of list elements with the given key
    void find( const Key& k, std::vector<Element*>& result )
    {
        Element* e = _first;
        while( e )
        {
            if( e->key==k ) result.push_back(e);
            else if( e->key>k ) break;
            else /* e->key<k */ e = e->next;
        }
    }

    /// add the given element
    void add( const Key& k, const T& value )
    {
        if( !_first )
            _first = new Element( k, value );
        else
        {
            Element* prev = _first;
            Element* cur = _first->next;
            while( cur && cur->key <= k )
            {
                prev = cur;
                cur = cur->next;
            }
            prev->next = new Element( k, value, cur );
        }
    }

    /// add the current element
    void del( Element* prev, Element* current )
    {
        assert( prev && current );
        Element* next = current->next;
        delete current;
        prev->next = next;
    }

    /// @returns the first list element pointer
    Element* first()  { return _first; }
    /// @returns the first list element const pointer
    const Element* first()  const { return _first; }

    /// perform a complete loop to count the element number
    size_t count() const
    {
        size_t t = 0;
        Element* e = _first;
        while( e )
        {
            ++t;
            e = e->next;
        }
        return t;
    }


protected:

    Element* _first; ///< the list first element

}; // class MultiMap_sortedList




} // namespace helper


} // namespace sofa


#endif // SOFA_NOPREALLOCATIONCONTAINERS_H
