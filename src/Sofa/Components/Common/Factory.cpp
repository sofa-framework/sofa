#include "Factory.inl"
#include <typeinfo>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace Sofa
{

namespace Components
{

namespace Common
{

/// Decode the type's name to a more readable form if possible
std::string gettypename(const std::type_info& t)
{
    std::string name = t.name();
#ifdef __GNUC__
    char* realname = NULL;
    int status;
    realname = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
    if (realname!=NULL)
    {
        name = realname;
        free(realname);
    }
#endif
    // Remove namespaces
    for(;;)
    {
        std::string::size_type pos = name.find("::");
        if (pos == std::string::npos) break;
        std::string::size_type first = name.find_last_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",pos-1);
        if (first == std::string::npos) first = 0;
        else first++;
        name.erase(first,pos-first+2);
    }
    //Remove "class "
    for(;;)
    {
        std::string::size_type pos = name.find("class ");
        if (pos == std::string::npos) break;
        name.erase(pos,6);
    }
    return name;
}

} // namespace Common

} // namespace Components

} // namespace Sofa
