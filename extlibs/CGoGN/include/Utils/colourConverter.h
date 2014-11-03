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

#ifndef __COLOURCONVERTER_H__
#define __COLOURCONVERTER_H__

#include <iostream>

#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"

// #define DISPLAY

namespace CGoGN {

namespace Utils {

/*!
 * \class ColourConverter
 * \brief Class for switching between different tri-channel color-spaces (see \link #ColourEncoding ColourEncoding enumeration\endlink  for available types)
 *
 * Usage:\n
 *  VEC3 colRGB ; 								// current colour in RGB for example\n
 *  ColourConverter<REAL> cc(colRGB,C_RGB) ;	// Tell constructor you provided RGB\n
 *  VEC3 colLuv = cc.getLuv() ; 				// ask whatever supported colour type you require
 *
 * Some conversion formulae were provided by "Notes about color", january 5th, 2011 by B. Sauvage
 */
template <typename REAL>
class ColourConverter
{
public: // types
	/**
	 * \enum ColourEncoding
	 * Supported colour spaces
	 */
	enum ColourEncoding
	{
		C_RGB = 0, /*!< R,G,B in [0,1] */
		C_XYZ = 1, /*!< X,Y,Z in [0,1] */
		C_Luv = 2, /*!< L in [0,100], u in [-83,175], v in [-134,108] */
		C_Lab = 3, /*!< L in [0,100], a in [-86,98], b in [-108,95] */
		C_HSV = 4,  /*!< H,S,V in [0,1] */
		C_HSL = 5  /*!< H,S,L in [0,1] */
	} ;

	typedef Geom::Vector<3,REAL> VEC3 ; /*!< Triplet for color encoding */

public: // methods
	/**
	 * \brief Constructor
	 * @param col a VEC3 colour
	 * @param enc the colour space of provided colour
	 */
	ColourConverter(const VEC3& col, const enum ColourEncoding& enc) ;
	/**
	 * \brief Destructor
	 */
	~ColourConverter() ;

	/**
	 * \brief getR
	 * @return original value (in its original space)
	 */
	VEC3 getOriginal() ;
	/**
	 * \brief getR
	 * @return value of provided colour
	 */
	VEC3 getColour(enum ColourEncoding enc) ;
	/**
	 * \brief getR
	 * @return RGB value of provided colour
	 */
	VEC3 getRGB() ;
	/**
	 * getR
	 * @return Luv value of provided colour
	 */
	VEC3 getLuv() ;
	/**
	 * getR
	 * @return Lab value of provided colour
	 */
	VEC3 getLab() ;
	/**
	 * getR
	 * @return Lab value of provided colour
	 */
	VEC3 getHSV() ;
	/**
	 * getR
	 * @return HSL value of provided colour
	 */
	VEC3 getHSL() ;
	/**
	 * getR
	 * @return XYZ value of provided colour
	 */
	VEC3 getXYZ() ;

public: // members
	enum ColourEncoding originalEnc ;  /*!< Colour space of original (unaltered) data */

private: // private members
	VEC3 *RGB ;
	VEC3 *Luv ;
	VEC3 *Lab ;
	VEC3 *HSV ;
	VEC3 *XYZ ;
	VEC3 *HSL ;

	bool convert(enum ColourEncoding from, enum ColourEncoding to) ;
	void convertRGBtoXYZ() ;
	void convertXYZtoRGB() ;

	void convertXYZtoLuv() ;
	void convertLuvToXYZ() ;

	void convertXYZtoLab() ;
	void convertLabToXYZ() ;

	/**
	 * Converts RGB to HSV. All is normalized between 0 and 1.
	 * Conversion formula adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 */
	void convertRGBtoHSV() ;
	/**
	 * Converts HSV to RGB. All is normalized between 0 and 1.
	 * Conversion formula adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 */
	void convertHSVtoRGB() ;

	/**
	 * Converts RGB to HSL. All is normalized between 0 and 1.
	 * Conversion formula adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 */
	void convertRGBtoHSL() ;
	/**
	 * Converts HSL to RGB. All is normalized between 0 and 1.
	 * Conversion formula adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 */
	void convertHSLtoRGB() ;

	static REAL hue2rgb(const REAL& p, const REAL& q, REAL t) ;

private: // private constants
	// D65 reference white
	static constexpr REAL Xn = 0.950456 ;
	static constexpr REAL Yn = 1.0 ;
	static constexpr REAL Zn = 1.088754 ;

	static constexpr REAL un = 0.197832 ;
	static constexpr REAL vn = 0.468340 ;

} ;

} // namespace Utils

} // namespace CGoGN

#include "Utils/colourConverter.hpp"

#endif // __COLOURCONVERTER_H__
