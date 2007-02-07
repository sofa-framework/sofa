#include <math.h>
#include <iostream>
#include "Quat.inl"

namespace Sofa
{

namespace Components
{

namespace Common
{

// instanciate the classes
template class Quater<double>;
template class Quater<float>;

// instanciate the friend methods
//template std::ostream& operator<<(std::ostream& out, Quater<float> Q);
//template std::ostream& operator<<(std::ostream& out, Quater<double> Q);

} // namespace Common

} // namespace Components

} // namespace Sofa
