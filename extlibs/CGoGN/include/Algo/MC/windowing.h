/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef WINDOWING_H
#define WINDOWING_H

#include "Algo/MC/type.h"

#include <iostream>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

/**
 * Common windowing class
 * @param DataType the type of image voxel
 *
 * Windowing derived  classes are used as template,
 * and minimum interface are function
 * - inside
 * - insideWich
 * - interpole
 *
 */
template<class DataType>
class WindowingClass
{
protected:
	/// value that cut space in two lower is outside upper is inside
	DataType m_value;

	/// min value (lower values are outside object
	DataType m_min;

	/// max value (upper values are outside object
	DataType m_max;

public:
	/**
	 * Default constructor
	 */
	WindowingClass() : m_value(), m_min(), m_max() {}

	/**
	 * copy constructor
	 */
	WindowingClass(const WindowingClass& wind) : m_value(wind.m_value), m_min(wind.m_min), m_max(wind.m_max) {}

	/**
	 * set the iso value to use
	 * @param val the isovalue
	 */
	void setIsoValue(DataType val) {
		m_value = val;
	}

	/**
	 * set the min and max  value to use
	 * @param min the min value
	 * @param max the max value
	 */
	void setMinMax(DataType min, DataType max) {
		m_min = min;
		m_max = max;
	}
};

/**
 * Windowing class: inside is only voxel with value equal to isoValue
 * @param DataType the type of image voxel
 */
template<class DataType>
class WindowingEqual : public WindowingClass<DataType>
{
public:

	WindowingEqual(){}

	/**
	 * copy constructor
	 */
	WindowingEqual(const WindowingEqual& wind) : WindowingClass<DataType>(wind) {}

	/**
	 * @return a label corresponding to value
	 * @param val voxel value
	 * @return label
	 */
	int16 insideWich(DataType val) const {
		return static_cast<int16>(val);
	}

	/**
	 * @return true if voxel is inside the object
	 */
	bool inside(DataType val) const {
		return val == this->m_value;
	}

	/**
	 * Give interpolation between to voxel value. Here always 0.5
	 * @param val1 voxel first value
	 * @param val2 voxel second value
	 */
	float interpole(DataType val1, DataType val2) const {
		return 0.5f;
	}
};

/**
 * Windowing class: inside is only voxel with value different of isoValue
 * @param DataType the type of image voxel
 */
template<class DataType>
class WindowingDiff : public WindowingClass<DataType>
{
public:

	WindowingDiff(){}

	/**
	 * copy constructor
	 */
	WindowingDiff(const WindowingDiff& wind) : WindowingClass<DataType>(wind) {}

	/**
	 * @return a label corresponding to value
	 * @param val voxel value
	 * @return label
	 */
	int16 insideWich(DataType val) const {
		return static_cast<int16>(val);
	}

	/**
	 * @return true if voxel is inside the object
	 */
	bool inside(DataType val) const {
		return val != this->m_value;
	}

	/**
	 * Give interpolation between to voxel value. Here always 0.5
	 * @param val1 voxel first value
	 * @param val2 voxel second value
	 */
	float interpole(DataType, DataType) const {
		return 0.5f;
	}
};

/**
 * Windowing class: inside is only voxel with value greater than of isoValue
 * @param DataType the type of image voxel
 */
template<class DataType>
class WindowingGreater : public WindowingClass<DataType>
{
public:

	WindowingGreater() {}

	/**
	 * copy constructor
	 */
	WindowingGreater(const WindowingGreater& wind) : WindowingClass<DataType>(wind) {}

	/**
	 * @return a label corresponding to value
	 * @param val voxel value
	 * @return label
	 */
	int16 insideWich(DataType val) const {
		return static_cast<int16>(val);
	}

	/**
	 * @return true if voxel is inside the object
	 */
	bool inside(DataType val) const {
		return  (val >= this->m_value);
	}

	/**
	 * Give interpolation between to voxel value
	 * @param val1 voxel first value
	 * @param val2 voxel second value
	 */
	float interpole(DataType val1, DataType val2) const {
		return  static_cast<float>(this->m_value - val1) / static_cast<float>(val2 - val1);
	}
};

/**
 * Windowing class: inside is only voxel with value greater than of isoValue
 * @param DataType the type of image voxel
 */
template<class DataType>
class WindowingLess : public WindowingClass<DataType>
{
public:

	WindowingLess() {}

	/**
	 * copy constructor
	 */
	WindowingLess(const WindowingLess& wind) : WindowingClass<DataType>(wind) {}

	/**
	 * @return a label corresponding to value
	 * @param val voxel value
	 * @return label
	 */
	int16 insideWich(DataType val) const {
		return static_cast<int16>(val);
	}

	/**
	 * @return true if voxel is inside the object
	 */
	bool inside(DataType val) const {
		return  (val<=this->m_value);
	}

	/**
	 * Give interpolation between to voxel value
	 * @param val1 voxel first value
	 * @param val2 voxel second value
	 */
	float interpole(DataType val1, DataType val2) const {
		return  static_cast<float>(this->m_value - val1) / static_cast<float>(val2 - val1);
	}
};


/**
 * Windowing class: inside is only voxel with value between min and max
 * @param DataType the type of image voxel
 */
template<class DataType>
class WindowingInterval : public WindowingClass<DataType>
{
public:

	WindowingInterval() {}

	/**
	 * copy constructor
	 */
	WindowingInterval(const WindowingInterval& wind) : WindowingClass<DataType>(wind) {}

	/**
	 * @return a label corresponding to value
	 * @param val voxel value
	 * @return label
	 */
	int16 insideWich(DataType val) const {
		return static_cast<int16>(val);
	}

	/**
	 * @return true if voxel is inside the object
	 */
	bool inside(DataType val) const {
		return (val>=this->m_min) && (val<=this->m_max);
	}

	/**
	 * Give interpolation between to voxel value
	 * @param val1 voxel first value
	 * @param val2 voxel second value
	 */
	float interpole(DataType val1, DataType val2) const {
		if (val1 < val2)
		{
			if (val1 < this->m_min)
			{
				return static_cast<float>(this->m_min - val1) / static_cast<float>(val2 - val1);
			}
			return static_cast<float>(this->m_max - val1) / static_cast<float>(val2 - val1);
		}

		if (val2 < this->m_min)
		{
			return static_cast<float>(this->m_min - val2) / static_cast<float>(val1 - val2);
		}
		return static_cast<float>(this->m_max - val2) / static_cast<float>(val1 - val2);
	}
};

}
} // end namespace
} // end namespace
} // end namespace

#endif


