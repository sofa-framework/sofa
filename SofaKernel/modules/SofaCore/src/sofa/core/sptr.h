#ifndef SOFA_CORE_SPTR_H
#define SOFA_CORE_SPTR_H

#include <boost/intrusive_ptr.hpp>

namespace sofa {

namespace core {

template<class T>
using sptr = boost::intrusive_ptr<T>;

}
}
 

#endif
