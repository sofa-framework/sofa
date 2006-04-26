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
template std::ostream& operator<<(std::ostream& out, Quater<float> Q);
template std::ostream& operator<<(std::ostream& out, Quater<double> Q);

//template Quater<float> operator+(Quater<float> q1, Quater<float> q2);
//template Quater<double> operator+(Quater<double> q1, Quater<double> q2);

//template Quater<float> operator*(const Quater<float>& q1, const Quater<float>& q2);
//template Quater<double> operator*(const Quater<double>& q1, const Quater<double>& q2);

} // namespace Common

} // namespace Components

} // namespace Sofa
