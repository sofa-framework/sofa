/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

template <typename PFP, typename T, unsigned int ORBIT>
Approximator<PFP,T,ORBIT>::Approximator(MAP& m, std::vector<VertexAttribute<T, MAP>* > va, Predictor<PFP, T> * predictor) :
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
		m_approx[i] = this->m_map.template addAttribute<T, ORBIT, MAP>(aname.str()) ;

		if(m_predictor)	// if predictors are associated to the approximator
		{				// create attributes to store the details needed for reconstruction
			std::stringstream dname ;
			dname << "detail_" << m_attrV[i]->name() ;
			m_detail[i] = this->m_map.template addAttribute<T, ORBIT, MAP>(dname.str()) ;
		}
	}
}

template <typename PFP, typename T, unsigned int ORBIT>
Approximator<PFP,T,ORBIT>::~Approximator()
{
//	std::cout << "Approximator<PFP,T,ORBIT>::~Approximator()" << std::endl ;
	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
	{
		this->m_map.removeAttribute(m_approx[i]) ;
		if(m_predictor)
			this->m_map.removeAttribute(m_detail[i]) ;
	}
}

template <typename PFP, typename T, unsigned int ORBIT>
const std::string&
Approximator<PFP,T,ORBIT>::getApproximatedAttributeName(unsigned int index) const
{
	return m_attrV[index]->name() ;
}

template <typename PFP, typename T, unsigned int ORBIT>
unsigned int
Approximator<PFP,T,ORBIT>::getNbApproximated() const
{
	return m_attrV.size() ;
}

template <typename PFP, typename T, unsigned int ORBIT>
void
Approximator<PFP,T,ORBIT>::saveApprox(Dart d)
{
	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
	{
		m_app[i] = m_approx[i][d] ;
	}
}

template <typename PFP, typename T, unsigned int ORBIT>
void
Approximator<PFP,T,ORBIT>::affectApprox(Dart d)
{
	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
	{
		m_attrV[i]->operator[](d) = m_app[i] ;
	}
}

template <typename PFP, typename T, unsigned int ORBIT>
const T&
Approximator<PFP,T,ORBIT>::getApprox(Dart d, unsigned int index) const
{
	return m_approx[index][d] ;
}

template <typename PFP, typename T, unsigned int ORBIT>
const VertexAttribute<T, typename PFP::MAP>&
Approximator<PFP,T,ORBIT>::getAttr(unsigned int index) const
{
	return *(m_attrV[index]) ;
}

template <typename PFP, typename T, unsigned int ORBIT>
VertexAttribute<T, typename PFP::MAP>&
Approximator<PFP,T,ORBIT>::getAttr(unsigned int index)
{
	return *(m_attrV[index]) ;
}

template <typename PFP, typename T, unsigned int ORBIT>
std::vector<T>
Approximator<PFP,T,ORBIT>::getAllApprox(Dart d) const
{
	std::vector<T> res ;
	res.resize(m_attrV.size()) ;
	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
		res[i] = m_approx[i][d] ;

	return res ;
}

template <typename PFP, typename T, unsigned int ORBIT>
const Predictor<PFP, T>*
Approximator<PFP,T,ORBIT>::getPredictor() const
{
	return m_predictor ;
}

template <typename PFP, typename T, unsigned int ORBIT>
const T&
Approximator<PFP,T,ORBIT>::getDetail(Dart d, unsigned int index) const
{
	assert(m_predictor || !"Trying to get detail on a non-predictive scheme") ;
	return m_detail[index][d] ;
}

template <typename PFP, typename T, unsigned int ORBIT>
std::vector<T>
Approximator<PFP,T,ORBIT>::getAllDetail(Dart d) const
{
	assert(m_predictor || !"Trying to get detail on a non-predictive scheme") ;

	std::vector<T> res ;
	res.resize(m_attrV.size()) ;
	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
		res[i] = m_detail[i][d] ;
	return res ;
}

template <typename PFP, typename T, unsigned int ORBIT>
void
Approximator<PFP,T,ORBIT>::setDetail(Dart d, unsigned int index, T& val)
{
	assert(m_predictor || !"Trying to set detail on a non-predictive scheme") ;
	m_detail[index][d] = val ;
}

template <typename PFP, typename T, unsigned int ORBIT>
void
Approximator<PFP,T,ORBIT>::setDetail(Dart d, std::vector<T>& val)
{
	assert(m_predictor || !"Trying to set detail on a non-predictive scheme") ;

	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
		m_detail[index][d] = val[i] ;
}


//	// TODO works only for vector types !!
//	REAL detailMagnitude(Dart d)
//	{
//		assert(m_predictor || !"Trying to get detail magnitude on a non-predictive scheme") ;
//		return m_detail[d].norm2() ;
//	}

template <typename PFP, typename T, unsigned int ORBIT>
void
Approximator<PFP,T,ORBIT>::addDetail(Dart d, double amount, bool sign, typename PFP::MATRIX33* detailTransform)
{
	assert(m_predictor || !"Trying to add detail on a non-predictive scheme") ;

	for (unsigned int i = 0 ; i < m_attrV.size() ; ++i)
	{
		T det = m_detail[i][d] ;
		if(detailTransform)
			det = (*detailTransform) * det ;
		det *= amount ;
		if(!sign)
			det *= REAL(-1) ;
		m_attrV[i]->operator[](d) += det ;
	}
}

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
