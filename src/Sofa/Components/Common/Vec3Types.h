#ifndef SOFA_COMPONENTS_COMMON_VEC3TYPES_H
#define SOFA_COMPONENTS_COMMON_VEC3TYPES_H

#include "Vec.h"
#include <Sofa/Components/Common/vector.h>
using Sofa::Components::Common::vector;
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class StdVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;

    /*!
     Basic Constraint applied to a single Degree Of Freedom (DOF)
    */
    class DOFConst
    {
    public:
        /*!
         The Construction needs the direction of the constraint and the index of the DOF
         on which the constraint must be applied
         This structure allows the management of sparse matrices of constraints
        */
        DOFConst(unsigned int index, Deriv& direction) : index(index), direction(direction) {}

        //! Index of the constrained DOF
        /*!
         Index of the corresponding DOF in the MechanicalModel State Vectors
        */
        unsigned int index;

        //! Direction Vector of the constraint
        Deriv direction;
    };

    //! A Single Constraint applied to the whole state Vector
    typedef vector<DOFConst> TConst;

    //! All the Constraints applied to a state Vector
    typedef	vector<TConst> VecConst;


    static void set(Coord& c, double x, double y, double z)
    {
        c[0] = (typename Coord::value_type)x;
        c[1] = (typename Coord::value_type)y;
        c[2] = (typename Coord::value_type)z;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = (double) c[0];
        y = (double) c[1];
        z = (double) c[2];
    }

    static void add(Coord& c, double x, double y, double z)
    {
        c[0] += (typename Coord::value_type)x;
        c[1] += (typename Coord::value_type)y;
        c[2] += (typename Coord::value_type)z;
    }
};

template<class T>
class ExtVector
{
public:
    typedef T              value_type;
    typedef unsigned int   size_type;

protected:
    value_type* data;
    size_type   maxsize;
    size_type   cursize;

public:
    ExtVector() : data(NULL), maxsize(0), cursize(0) {}
    virtual ~ExtVector() {}
    void setData(value_type* d, size_type s) { data=d; maxsize=s; cursize=s; }
    value_type& operator[](size_type i) { return data[i]; }
    const value_type& operator[](size_type i) const { return data[i]; }
    size_type size() const { return cursize; }
    bool empty() const { return cursize==0; }
    virtual void resize(size_type size)
    {
        if (size <= maxsize)
            cursize = size;
        else
        {
            cursize = maxsize;
            std::cerr << "Error: invalide resize request ("<<size<<">"<<maxsize<<") on external vector.\n";
        }
    }
};

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class ExtVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef ExtVector<Coord> VecCoord;
    typedef ExtVector<Deriv> VecDeriv;

    /*!
     Basic Constraint applied to a single Degree Of Freedom (DOF)
    */
    class DOFConst
    {
    public:
        /*!
         The Construction needs the direction of the constraint and the index of the DOF
         on which the constraint must be applied
         This structure allows the management of sparse matrices of constraints
        */
        DOFConst(unsigned int index, Deriv& direction) : index(index), direction(direction) {}

        //! Index of the constrained DOF
        /*!
         Index of the corresponding DOF in the MechanicalModel State Vectors
        */
        unsigned int index;

        //! Direction Vector of the constraint
        Deriv direction;
    };

    //! A Single Constraint applied to the whole state Vector
    typedef vector<DOFConst> TConst;

    //! All the Constraints applied to a state Vector
    typedef	vector<TConst> VecConst;


    static void set(Coord& c, double x, double y, double z)
    {
        c[0] = (typename Coord::value_type)x;
        c[1] = (typename Coord::value_type)y;
        c[2] = (typename Coord::value_type)z;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = (double) c[0];
        y = (double) c[1];
        z = (double) c[2];
    }

    static void add(Coord& c, double x, double y, double z)
    {
        c[0] += (typename Coord::value_type)x;
        c[1] += (typename Coord::value_type)y;
        c[2] += (typename Coord::value_type)z;
    }
};

typedef StdVectorTypes<Vec3d,Vec3d,double> Vec3dTypes;
typedef StdVectorTypes<Vec3f,Vec3f,float> Vec3fTypes;
typedef Vec3dTypes Vec3Types;

typedef ExtVectorTypes<Vec3d,Vec3d,double> ExtVec3dTypes;
typedef ExtVectorTypes<Vec3f,Vec3f,float> ExtVec3fTypes;
typedef Vec3dTypes ExtVec3Types;

} // namespace Common

} // namespace Components

} // namespace Sofa


#endif
