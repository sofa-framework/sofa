#ifndef SOFA_CORE_TYPETRAITS_H
#define SOFA_CORE_TYPETRAITS_H
namespace sofa {
namespace core {

#include <type_traits>

////////////////////////////////////////////////////////////////////////////////////////
/// A static type conversion.
/// The problem with other casts is that they assumes the type hierarchy is known.
/// This can be problematic when manipulating class from their forward declaration.
/// So a class that want to expose some casting capability can implement this
/// function.
/// Eg:
///      class MyClass ;
///      template<>
///       inline ParentClass As(MyClass* c){ return reinterpret_cast<ParentClass>(*p); }
///
/// This function can be dangerous if suddenly an object once inhereting from another
/// has is inheritence graph changed. So use this function with care.
///
////////////////////////////////////////////////////////////////////////////////////////
/*template<class T, class C, typename std::enable_if<std::is_same<C, T>::value, int>::type = 0  >
inline T* As(C*)
{
    static_assert(std::is_same<C, C>::value, "You should implement the As conversion") ;
    return nullptr ;
}*/

}
}
 
#endif
