#include <math.h>
#include <iostream>
#include <sofa/defaulttype/Quat.inl>

namespace sofa
{

namespace defaulttype
{

// instanciate the classes
template class Quater<double>;
template class Quater<float>;

// instanciate the friend methods
//template std::ostream& operator<<(std::ostream& out, Quater<float> Q);
//template std::ostream& operator<<(std::ostream& out, Quater<double> Q);

} // namespace defaulttype

} // namespace sofa

