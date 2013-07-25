#ifndef PAIR_H
#define PAIR_H

#include <ostream>
#include <istream>

namespace sofa {

namespace helper
{

template<class First, class Second>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<First, Second>& p)
{
	return out << p.first << " " << p.second;
}


template<class First, class Second>
std::istream& operator>>(std::istream& in,
                         std::pair<First, Second>& p)
{
	return in >> p.first >> p.second;
}

}
}

#endif
