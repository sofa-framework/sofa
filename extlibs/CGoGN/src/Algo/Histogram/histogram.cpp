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

#include "Algo/Histogram/histogram.h"


namespace CGoGN
{

namespace Algo
{

namespace Histogram
{


void Histogram::initDataConvert(const AttributeConvertGen& conv)
{
	unsigned int beg = conv.begin();
	m_min = conv[beg];
	m_max = m_min;

	m_dataIdx.reserve(conv.nbElements());
	for (unsigned int i = beg; i!= conv.end(); conv.next(i))
	{
		double val = conv[i];
		m_dataIdx.push_back(std::make_pair(val,i));
		if (val < m_min)
			m_min = val;
		if (val > m_max)
			m_max = val;
	}

	m_hcolmap.setMin(m_min);
	m_hcolmap.setMax(m_max);
}

void Histogram::populateHisto(unsigned int nbclasses)
{
	//compute nb classes if necesary
	if (nbclasses == 0)
		m_nbclasses = (unsigned int)(sqrt(double(m_dataIdx.size())));
	else
		m_nbclasses = nbclasses;

	m_hcolmap.setNb(m_nbclasses);

	//compute width interv
	m_interWidth = (m_max-m_min)/double(m_nbclasses);

	// init to zero
	m_populations.resize(m_nbclasses);
	for (unsigned int i = 0; i<m_nbclasses; ++i)
		m_populations[i] = 0;

	// traverse attribute to populate
	for (std::vector<std::pair<double,unsigned int> >::const_iterator it = m_dataIdx.begin(); it != m_dataIdx.end(); ++it)
	{
		unsigned int c = whichClass(it->first);
		if (c != 0xffffffff)
			m_populations[c]++;
	}
    m_maxBar = 0;
    for (unsigned int i = 0; i<m_nbclasses; ++i)
    {
    	if (m_populations[i] > m_maxBar)
    		m_maxBar = m_populations[i];
    }

    // apply area correction on quantile if necessary
    if (m_pop_quantiles.size() != 0 )
       	quantilesAreaCorrection();

}

void Histogram::populateQuantiles(unsigned int nbquantiles)
{
	if (!m_sorted)
	{
		std::sort(m_dataIdx.begin(),m_dataIdx.end(),dataComp);
		m_sorted = true;
	}

	// compute exact populations
	unsigned int nb = m_dataIdx.size();
	double pop = double(nb)/nbquantiles;
	m_pop_quantiles.resize(nbquantiles);

	for (unsigned int i = 0; i < nbquantiles; ++i)
		m_pop_quantiles[i]=pop;

	m_interv.clear();
	m_interv.reserve(nbquantiles+1);
	// quantiles computation
	m_interv.push_back(m_dataIdx.front().first);
	double cumul = 0.0;
	for (unsigned int i = 0; i < nbquantiles; ++i)
	{
		cumul += m_pop_quantiles[i];
		unsigned int icum = floor(cumul);
		double val = 0.0;
		if (icum < m_dataIdx.size()-1)
			val = (data(icum)+ data(icum+1)) / 2.0;
		else
			val = m_dataIdx.back().first;
		m_interv.push_back(val);
	}
	quantilesAreaCorrection();
}


void Histogram::quantilesAreaCorrection()
{
	unsigned int nbquantiles = m_pop_quantiles.size();

	// constant area correction
	double areaQ1 = 100.0f;	// use 100 as area if no histogram
	if (m_nbclasses!=0)
	{
		double areaH = (getMax()-getMin())/double(m_nbclasses) * double(m_dataIdx.size());
		areaQ1 = areaH/nbquantiles; // area of one quantile
	}

	m_maxQBar = 0.0;
	for (unsigned int i = 0; i < nbquantiles; ++i)
	{
		// compute height instead of population
		double lq = m_interv[i+1] - m_interv[i]; // width
		m_pop_quantiles[i] = areaQ1/lq;			// height = area / width
		if (m_pop_quantiles[i] > m_maxQBar)
			m_maxQBar = m_pop_quantiles[i];		// store max
	}
}


void Histogram::histoColorizeVBO(Utils::VBO& vbo)
{
	unsigned int nb = m_dataIdx.size();
	vbo.setDataSize(3);
	vbo.allocate(nb);
	Geom::Vec3f* colors = static_cast<Geom::Vec3f*>(vbo.lockPtr());
	for (unsigned int i = 0; i<nb; ++i)
	{
		unsigned int j = idx(i);
		unsigned int c = whichClass(data(i));
		if (c != 0xffffffff)
			colors[j] = m_hcolmap.colorIndex(c);
	}
	vbo.releasePtr();
}


unsigned int Histogram::cellsOfHistogramColumn(unsigned int c, std::vector<unsigned int>& vc) const
{
	if (!m_sorted)
	{
		std::sort(m_dataIdx.begin(),m_dataIdx.end(),dataComp);
		m_sorted = true;
	}

	vc.clear();

	double bi = (m_max-m_min)/m_nbclasses * c + m_min;
	double bs = (m_max-m_min)/m_nbclasses * (c+1) + m_min;

	unsigned int nb=m_dataIdx.size();
	unsigned int i=0;

	while ((i<nb) && (data(i)< bi))
		++i;

	while ((i<nb) && (data(i)< bs))
		vc.push_back(idx(i++));

	return vc.size();
}

unsigned int Histogram::cellsOfQuantilesColumn( unsigned int c, std::vector<unsigned int>& vc) const
{
	vc.clear();

	double bi = m_interv[c];
	double bs = m_interv[c+1];

	unsigned int nb=m_dataIdx.size();
	unsigned int i=0;

	while ((i<nb) && (data(i)< bi))
		++i;

	while ((i<nb) && (data(i)< bs))
		vc.push_back(idx(i++));

	return vc.size();
}


}
}
}


