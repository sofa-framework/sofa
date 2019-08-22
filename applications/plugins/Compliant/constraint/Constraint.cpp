#include "Constraint.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

size_t Constraint::s_lastConstraintTypeIndex = 0;

Constraint::Constraint() : mask( NULL ) {}


}
}
}
