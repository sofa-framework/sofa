#ifndef SOFA_COMPONENTS_COMMON_VEC3TYPES_H
#define SOFA_COMPONENTS_COMMON_VEC3TYPES_H

#include "Vec.h"
#include <vector>
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class TCoord, class TDeriv>
class StdVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef std::vector<Coord> VecCoord;
    typedef std::vector<Deriv> VecDeriv;
};

template<class T>
class ExtVector
{
public:
    typedef T              data_type;
    typedef unsigned int   size_type;

protected:
    data_type* data;
    size_type maxsize;
    size_type cursize;

public:
    ExtVector() : data(NULL), maxsize(0), cursize(0) {}
    void setData(data_type* d, size_type s) { data=d; maxsize=s; cursize=s; }
    data_type& operator[](size_type i) { return data[i]; }
    const data_type& operator[](size_type i) const { return data[i]; }
    size_type size() const { return cursize; }
    void resize(size_type size)
    {
        if (size <= maxsize)
            cursize = maxsize;
        else
        {
            cursize = maxsize;
            std::cerr << "Error: invalide resize request ("<<size<<">"<<maxsize<<" on external vector\n";
        }
    }
};

template<class TCoord, class TDeriv>
class ExtVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef ExtVector<Coord> VecCoord;
    typedef ExtVector<Deriv> VecDeriv;
};

typedef StdVectorTypes<Vec3d,Vec3d> Vec3dTypes;
typedef StdVectorTypes<Vec3f,Vec3f> Vec3fTypes;
typedef Vec3dTypes Vec3Types;

typedef ExtVectorTypes<Vec3d,Vec3d> ExtVec3dTypes;
typedef ExtVectorTypes<Vec3f,Vec3f> ExtVec3fTypes;
typedef Vec3dTypes ExtVec3Types;

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
