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

#ifndef __HISTOGRAM__
#define __HISTOGRAM__

#define _USE_MATH_DEFINES
#include <cmath>

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/cellmarker.h"
#include "Geometry/vector_gen.h"
#include "Utils/colorMaps.h"
#include "Utils/vbo_base.h"

namespace CGoGN
{

namespace Algo
{

namespace Histogram
{

class HistoColorMap
{
protected:
	double m_min;
	double m_max;
	double m_w;
	unsigned int m_nb;

public:
	virtual ~HistoColorMap() {}

	/// set min value (for update from Histogram)
	void setMin(double m) {m_min = m; m_w = (m_max -m_min)/double(m_nb);}

	/// set max value (for update from Histogram)
	void setMax(double m) {m_max = m; m_w = (m_max -m_min)/double(m_nb);}

	/// set nb value (for update from Histogram)
	void setNb(unsigned int n) {m_nb = n; m_w = (m_max -m_min)/double(m_nb);}

	/**
	 * get color from param: To implement (with call to colormap functions for examples)
	 */
	virtual Geom::Vec3f color(double v) const = 0 ;

	/**
	 * get color from index (can be overload)
	 * compute a [0,1[ value from an index [0,nb[
	 * and call color with it
	 * Used by Histogram::colorize && Qt::DrawHistogram
	 */
	virtual Geom::Vec3f colorIndex(unsigned int i) const
	{
		double v = double(i)/double(m_nb-1) + double(1)/double(m_nb+m_nb);
		return color(v);
	}
};

/**
 *  inherits this class by:
 *  - add xxAttribute& in data (& constructor)
 *  - overload begin/end/next/nbElements (by calling xxxAttribute.yyy)
 *  - overload operator [] to return the [i] converted in double with necessary computations.
 */
class AttributeConvertGen
{
public:
	virtual unsigned int begin() const = 0;
	virtual unsigned int end() const = 0;
	virtual void next(unsigned int& i) const = 0;
	virtual unsigned int nbElements() const = 0;
	virtual double operator[](unsigned int i) const = 0;
	virtual ~AttributeConvertGen() {}
};

/**
 * Helper class templated by Attribute
 * Avoid the writing of begin/end/next/nbElements
 */
template <typename ATT>
class AttributeConvert: public AttributeConvertGen
{
protected:
	ATT& attrib;

public:
	AttributeConvert(ATT &att): attrib(att) {}
	virtual unsigned int begin() const { return attrib.begin();}
	virtual unsigned int end() const { return attrib.end();}
	virtual void next(unsigned int& i) const { attrib.next(i);}
	virtual unsigned int nbElements() const { return attrib.nbElements();}
	virtual double operator[](unsigned int i) const = 0;
	virtual ~AttributeConvert() {}
};

/**
 * Histogram class
 * T must have operators -, / ,< ,>
 */
class Histogram
{
//	std::vector<double> m_data;

	mutable std::vector< std::pair<double, unsigned int> > m_dataIdx;
	
	/// number of classes in attribute
	unsigned int m_nbclasses;
	
	/// vector of population
	std::vector<unsigned int> m_populations;
	
	/// vector of intervals of quantiles
	std::vector<double> m_interv;

	/// vector of population for quantiles
	std::vector<double> m_pop_quantiles;

	/// min value
	double m_min;

	/// max value
	double m_max;

	/// interval width (in regular case)
	double m_interWidth;
	
	/// max number of population in a class
	unsigned int m_nbMin;

	/// max values in histo population
	unsigned int m_maxBar;

	/// max value in quantille population
	double m_maxQBar;

	HistoColorMap& m_hcolmap;

	mutable bool m_sorted;

	/// get data
	double data(unsigned int i) const;

	/// get idx of data in attribute
	unsigned int idx(unsigned int i) const;

	/// comparison function for sorting data
	static bool dataComp( const std::pair<double, unsigned int>& a, const std::pair<double, unsigned int>& b);

	/// update quantiles height from histo area for correct superposition
	void quantilesAreaCorrection();

public:
	/**
	* create an histogram from attribute handler
	*/
	Histogram(HistoColorMap& hcm);

	/**
	 * init data
	 * @param conv a attribute convertor
	 */
	void initDataConvert(const AttributeConvertGen& conv);

	/**
	 * init data
	 * @param attr the attribute to copy from
	 * @param sortForQuantiles sort data vector for quantille generation
	 */
	template <typename ATTR>
	void initData(const ATTR& attr);

	/**
	 * get min value of attribute (perhaps modified by user)
	 */
	double getMin() const;

	/**
	 * get max value of attribute (perhaps modified by user)
	 */
	double getMax() const;

	/**
	 * get real min value of attribute
	 */
	double getQMin() const;

	/**
	 * get real max value of attribute
	 */
	double getQMax() const;


	/**
	 * set min value of attribute
	 */
	void setMin(double m);

	/**
	 * set max value of attribute
	 */
	void setMax(double m);

	/**
	 * get max population value of all bars of histo
	 */
	unsigned int getMaxBar() const;

	/**
	 * get max population value of all bars of quantiles
	 */
	double getMaxQBar() const;


	/**
	 * modify min/max values to center Histogram on zero if necessary
	 */
	void centerOnZero();

	/**
	 * compute the histogram with given numbre of classes
	 */
	void populateHisto(unsigned int nbclasses = 0);

	/**
	 * compute the histogram with given number of classes
	 */
	void populateQuantiles(unsigned int nbclasses = 10);

	/**
	* which class belong a value
	*/
	unsigned int whichClass(double val) const;

	/**
	* which class belong a value
	*/
	unsigned int whichQuantille(double val) const;

	/**
	* fill a color attribute from histo
	* @param colors attribute to fill
	*/
	template <typename ATTC>
	void histoColorize(ATTC& colors);


	/**
	 * colorize the VBO (RGB) from histo
	 * @warning GL context must be accessible
	 * @param vbo the vbo to fill with colors
	 */
	void histoColorizeVBO(Utils::VBO& vbo);


	/**
	* fill a color attribute from quantiles
	* @param colors attribute to fill
	* @param tc table of color
	*/
	template<typename ATTC>
	void quantilesColorize(ATTC& colors, const std::vector<Geom::Vec3f>& tc);

	/**
	* colorize the VBO (RGB) from
	* @warning GL context must be accessible
	* @param vbo the vbo to fill with colors
	* @param tc table of color
	*/
	void quantilesColorizeVBO(Utils::VBO& vbo, const std::vector<Geom::Vec3f>& tc);

	/**
	* get the vector of class population
	*/
	const std::vector<unsigned int>& getPopulation() const;

	/**
	* get the vector of height of quantiles
	*/
	const std::vector<double>& getQuantilesHeights() const;

	/**
	* get the vector of intervals bounaries for quantiles
	*/
	const std::vector<double>& getQuantilesIntervals() const;

	/**
	 * return cells of histogram's column
	 * @param c column of histogram
	 * @param vc vector of cells (indices)
	 * @return number of cells
	 */
	unsigned int cellsOfHistogramColumn(unsigned int c, std::vector<unsigned int>& vc) const;

	/**
	 * return cells of quantile's column
	 * @param c column of quantile
	 * @param vc vector of cells (indices)
	 * @return number of cells
	 */
	unsigned int cellsOfQuantilesColumn(unsigned int c, std::vector<unsigned int>& vc) const;


	/**
	 * mark cells of histogram's column
	 * @param c column of quantile
	 * @param cm marker
	 * @return number of marked cells
	 */
	template <typename CELLMARKER>
	unsigned int markCellsOfHistogramColumn(unsigned int c, CELLMARKER& cm) const;

	/**
	 * mark cells of quantile's column
	 * @param c column of quantile
	 * @param cm marker
	 * @return number of marked cells
	 */
	template <typename CELLMARKER>
	unsigned int markCellsOfQuantilesColumn(unsigned int c, CELLMARKER& cm) const;

	/**
	 * get the colorMap
	 */
	const HistoColorMap& colorMap() const;
};

} // namespace Histogram

} // namespace Algo

} // namespace CGoGN

#include "Algo/Histogram/histogram.hpp"
#endif
