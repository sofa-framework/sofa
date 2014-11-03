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

#include <algorithm>

namespace CGoGN
{

namespace Algo
{

namespace Histogram
{

inline Histogram::Histogram( HistoColorMap& hcm):
 m_hcolmap(hcm),m_sorted(false)
{
}

inline const std::vector<unsigned int>& Histogram::getPopulation() const
{
	return m_populations;
}

inline const std::vector<double>& Histogram::getQuantilesHeights() const
{
	return m_pop_quantiles;
}

inline const std::vector<double>&  Histogram::getQuantilesIntervals() const
{
	return m_interv;
}

inline const HistoColorMap& Histogram::colorMap() const
{
	return m_hcolmap;
}

inline double Histogram::getMin() const
{
	return m_min;
}

inline double Histogram::getMax() const
{
	return m_max;
}

inline double Histogram::getQMin() const
{
	return m_dataIdx.front().first;
}

inline double Histogram::getQMax() const
{
	return m_dataIdx.back().first;
}

inline unsigned int Histogram::getMaxBar() const
{
	return m_maxBar;
}

inline double Histogram::getMaxQBar() const
{
	return m_maxQBar;
}

inline void Histogram::setMin(double m)
{
	m_min = m;
	m_hcolmap.setMin(m);
}

inline void Histogram::setMax(double m)
{
	m_max = m;
	m_hcolmap.setMax(m);
}

inline void Histogram::centerOnZero()
{
	if ((m_min <0.0) && (m_max > 0.0))
	{
		if ((-m_min) > m_max)
			m_max = -m_min;
		else
			m_min = -m_max;
	}
}

template <typename ATTR>
inline void Histogram::initData(const ATTR& attr)
{
	unsigned int beg = attr.begin();
	m_min = attr[beg];
	m_max = m_min;

	m_dataIdx.reserve(attr.nbElements());
	m_dataIdx.clear();
	for (unsigned int i = beg; i!= attr.end(); attr.next(i))
	{
		double val = attr[i];
		m_dataIdx.push_back(std::make_pair(val ,i));
		if (val < m_min)
			m_min = val;
		if (val > m_max)
			m_max = val;
	}

	m_hcolmap.setMin(m_min);
	m_hcolmap.setMax(m_max);

	m_sorted = false;
}

inline unsigned int Histogram::whichClass(double val) const
{
	if (val == m_max)
		return m_populations.size()-1;
	double x = (val - m_min)/m_interWidth;
	if ((x<0) || (val>=m_max))
		return -1;
	return (unsigned int)(x);
}

inline unsigned int Histogram::whichQuantille(double val) const
{
	unsigned int i=1;
	while (val > m_interv[i])
		++i;
	return i-1;
}

template<typename ATTC>
void Histogram::histoColorize(ATTC& colors)
{
	unsigned int nb = m_dataIdx.size();
	for (unsigned int i = 0; i<nb; ++i)
	{
		unsigned int j = idx(i);
		unsigned int c = whichClass(data(i));
		if (c != 0xffffffff)
			colors[j] = m_hcolmap.colorIndex(c);
	}
}

template<typename ATTC>
void Histogram::quantilesColorize(ATTC& colors, const std::vector<Geom::Vec3f>& tc)
{

	unsigned int nb = m_dataIdx.size();
	unsigned int nbi = m_interv.size()-1;

	assert(tc.size() >= nbi);

	unsigned int i=0;
	unsigned int j=0;

	while ((i<nb) && (j<nbi))
	{
		while ((i<nb) && (data(i) <= m_interv[j+1]))
			colors[ idx(i++) ] = tc[j];
		++j;
	}
}

/// get data
inline double Histogram::data(unsigned int i) const
{
	return m_dataIdx[i].first;
}

/// get idx of data in attribute
inline unsigned int Histogram::idx(unsigned int i) const
{
	return m_dataIdx[i].second;
}

// comparison function for sorting data
inline bool Histogram::dataComp( const std::pair<double, unsigned int>& a, const std::pair<double, unsigned int>& b)
{
	return a.first < b.first;
}

template <typename CELLMARKER>
unsigned int Histogram::markCellsOfHistogramColumn(unsigned int c, CELLMARKER& cm) const
{
	if (!m_sorted)
	{
		std::sort(m_dataIdx.begin(),m_dataIdx.end(),dataComp);
		m_sorted = true;
	}

	double bi = (m_max-m_min)/m_nbclasses * c + m_min;
	double bs = (m_max-m_min)/m_nbclasses * (c+1) + m_min;

	unsigned int nb=m_dataIdx.size();
	unsigned int i=0;

	while ((i<nb) && (data(i)< bi))
		++i;

	unsigned int nbc=0;
	while ((i<nb) && (data(i)< bs))
	{
		cm.mark(idx(i++));
		++nbc;
	}

	return nbc;
}

template <typename CELLMARKER>
unsigned int Histogram::markCellsOfQuantilesColumn(unsigned int c, CELLMARKER& cm) const
{
	double bi = m_interv[c];
	double bs = m_interv[c+1];

	unsigned int nb=m_dataIdx.size();
	unsigned int i=0;

	while ((i<nb) && (data(i)< bi))
		++i;

	unsigned int nbc=0;
	while ((i<nb) && (data(i)< bs))
	{
		cm.mark(idx(i++));
		++nbc;
	}

	return nbc;
}

} // namespace Histogram

} // namespace Algo

} // namespace CGoGN
