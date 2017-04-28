#ifndef SOFA_HELPER_TEMPLATE_NAME_H
#define SOFA_HELPER_TEMPLATE_NAME_H

#include <sstream>

namespace sofa {
namespace helper {

template< template<class ...> class T, class ... Args>
static std::string template_name(const T<Args...>*) {
    std::stringstream ss;
        
    const int expand[] = {
        (ss << Args::Name() << ',' , 0)...
    }; (void) expand;
        
    std::string res = ss.str();
    res.pop_back();
    return res;
}

}
}

#endif
