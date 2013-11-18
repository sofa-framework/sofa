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

#ifndef __APPROXIMATOR_VOLUMES_H__
#define __APPROXIMATOR_VOLUMES_H__

#include "Algo/DecimationVolumes/operator.h"
#include "Algo/DecimationVolumes/predictor.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

enum ApproximatorType
{
	A_QEM,
	A_MidEdge,
	A_MidFace,
	A_MidVolume,
	A_hHalfEdgeCollapse,
	A_QEM
};

template <typename PFP>
class ApproximatorGen
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	MAP& m_map ;

public:
	ApproximatorGen(MAP& m) : m_map(m)
	{}
	virtual ~ApproximatorGen()
	{}
	virtual const std::string& getApproximatedAttributeName(unsigned int index = 0) const = 0 ;
	virtual ApproximatorType getType() const = 0 ;
	virtual bool init() = 0 ;
	virtual void approximate(Dart d) = 0 ;
	virtual void saveApprox(Dart d) = 0 ;
	virtual void affectApprox(Dart d) = 0 ;
	virtual const PredictorGen<PFP>* getPredictor() const = 0 ;
} ;


template <typename PFP, typename T, unsigned int ORBIT>
class Approximator :  public ApproximatorGen<PFP>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL;

protected:
	Predictor<PFP, T>* m_predictor ;

	std::vector<VertexAttribute<T>* > m_attrV ;	// vertex attributes to be approximated
	std::vector<AttributeHandler<T,ORBIT> > m_approx ;	// attributes to store approximation result
	std::vector<AttributeHandler<T,ORBIT> > m_detail ;	// attributes to store detail information for reconstruction
	std::vector<T> m_app ;

public:
	Approximator(MAP& m, std::vector<VertexAttribute<T>* > va, Predictor<PFP, T> * predictor) :
		ApproximatorGen<PFP>(m), m_predictor(predictor), m_attrV(va)
	{
		const unsigned int& size = m_attrV.size() ;
		assert(size > 0 || !"Approximator: no attributes provided") ;

		m_approx.resize(size) ;
		m_detail.resize(size) ;
		m_app.resize(size) ;

		for (unsigned int i = 0 ; i < size ; ++i)
		{
			if (!m_attrV[i]->isValid())
				std::cerr << "Approximator Warning: attribute number " << i << " is not valid" << std::endl ;

			std::stringstream aname ;
			aname << "approx_" << m_attrV[i]->name() ;
			m_approx[i] = this->m_map.template addAttribute<T, ORBIT>(aname.str()) ;

			if(m_predictor)	// if predictors are associated to the approximator
			{				// create attributes to store the details needed for reconstruction
				std::stringstream dname ;
				dname << "detail_" << m_attrV[i]->name() ;
				m_detail[i] = this->m_map.template addAttribute<T, ORBIT>(dname.str()) ;
			}
		}
	}

	virtual ~Approximator()
	{
		for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
		{
			this->m_map.template removeAttribute(m_approx[i]) ;
			if(m_predictor)
				this->m_map.template removeAttribute(m_detail[i]) ;
		}
	}

	const std::string& getApproximatedAttributeName(unsigned int index = 0) const
	{
		return m_attrV[index]->name() ;
	}

	unsigned int getNbApproximated() const
	{
		return m_attrV.size() ;
	}

	void saveApprox(Dart d)
	{
		for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
			m_app[i] = m_approx[i][d] ;
	}

	void affectApprox(Dart d)
	{
		for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
			m_attrV[i]->operator[](d) = m_app[i] ;
	}

	const T& getApprox(Dart d, unsigned int index = 0) const
	{
		return m_approx[index][d] ;
	}

	const VertexAttribute<T>& getAttr(unsigned int index = 0) const
	{
		return *(m_attrV[index]) ;
	}

	std::vector<T> getAllApprox(Dart d) const
	{
		std::vector<T> res ;
		res.resize(m_attrV.size()) ;
		for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
			res[i] = m_approx[i][d] ;

		return res ;
	}

	const Predictor<PFP, T>* getPredictor() const
	{
		return m_predictor ;
	}

};


} // namespace Decimation

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#endif
