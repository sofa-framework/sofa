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

#ifndef __PREDICTOR_H__
#define __PREDICTOR_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

enum PredictorType
{
	P_CornerCutting,
	P_TangentPredict1,
	P_TangentPredict2,
	P_HalfCollapse
} ;

template <typename PFP>
class PredictorGen
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	MAP& m_map ;

public:
	PredictorGen(MAP& m) : m_map(m)
	{}
	virtual ~PredictorGen()
	{}
	virtual const std::string& getPredictedAttributeName() = 0 ;
	virtual PredictorType getType() = 0 ;
	virtual bool init() = 0 ;
	virtual void predict(Dart d2, Dart dd2) = 0 ;
	virtual void affectPredict(Dart d) = 0 ;
} ;


template <typename PFP, typename T>
class Predictor : public PredictorGen<PFP>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	VertexAttribute<T, MAP>& m_attrV ;	// vertex attribute to be predicted
	std::vector<T> m_predict ;		// prediction results

public:
	Predictor(MAP& m, VertexAttribute<T, MAP>& p) :
		PredictorGen<PFP>(m), m_attrV(p)
	{}

	virtual ~Predictor()
	{}

	const std::string& getPredictedAttributeName()
	{
		return m_attrV.name() ;
	}

	T& getPredict(unsigned int index)
	{
		return m_predict[index] ;
	}

	void affectPredict(Dart d)
	{
		Dart dd = this->m_map.phi2(d) ;
		m_attrV[d] = m_predict[0] ;
		m_attrV[dd] = m_predict[1] ;
	}
} ;

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
