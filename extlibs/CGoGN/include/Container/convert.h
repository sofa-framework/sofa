#ifndef _CONVERT_H_
#define _CONVERT_H_

#include <limits>
#include <math.h>
#include <iostream>
#include "Geometry/vector_gen.h"

namespace CGoGN
{
namespace DataConversion
{
/**
 * function that convert a Vec of double into a Vec of float
 * template param D: dimension of vector
 */
template< unsigned int D>
inline typename Geom::Vector<D,float>::type funcVecXdToVecXf(const typename Geom::Vector<D,double>::type & x)
{
    typename Geom::Vector<D,float>::type v;
	for (unsigned int j=0; j<D; ++j)
		v[j] = float(x[j]);
	return v;
}

/**
 * function that convert a scalar (char/short/int/double) into float
 * template param TYPE_IN: scalar type to convert
 */
template<typename TYPE_IN>
inline float funcToFloat(const TYPE_IN& x)
{
	return float(x);
}

/**
 * functor that convert a scalar (char/short/int/double) into float normalized in [0,1]
 * template param TYPE_IN: scalar type to convert
 * Constructor params: min/max values of data for normalization
 */
template<typename TYPE_IN>
class functorToFloatNormalized
{
protected:
	double m_min;
	double m_diff;
public:
	functorToFloatNormalized(TYPE_IN min, TYPE_IN max) :
		m_min(double(min)),
		m_diff(double(max-min)) {}

	inline float operator() (const TYPE_IN& x)
	{
		double v = (double(x) - m_min)/ m_diff;
		return float(v);
	}
};

/**
 * functor that convert a scalar (char/short/int/double) into a RGB color (Vec3f)
 * template param TYPE_IN: scalar type to convert
 * Constructor params: min/max values of data for normalization
 */
template<typename TYPE_IN>
class functorScalarToRGBf
{
protected:
	double m_min;
	double m_diff;
public:
	functorScalarToRGBf(TYPE_IN min, TYPE_IN max) :
		m_min(double(min)),
		m_diff(double(max-min)) {}

	inline Geom::Vec3f operator() (const TYPE_IN& x)
	{
		double h = (360.0 /m_diff) * (double(x) - m_min); // normalize in 0-360
		int hi = int(floor(h / 60.0)) % 6;
		float f = float((h / 60.0) - floor(h / 60.0));
		float q = 1.0f - f;
		switch(hi)
		{
			case 0:
			return Geom::Vec3f(0.0f,f,1.0f);
				break;
			case 1:
			return Geom::Vec3f(0.0f,1.0f,q);
				break;
			case 2:
			return Geom::Vec3f(f,1.0f,0.0f);
				break;
			case 3:
			return Geom::Vec3f(1.0f,q,0.0f);
				break;
			case 4:
			return Geom::Vec3f(1.0f,0.0f,f);
				break;
			case 5:
			return Geom::Vec3f(q,0.0f,1.0f);
			default:
				break;
		}
		return Geom::Vec3f(0.0f,0.0f,0.0f);
	}
};

}

}

#endif
