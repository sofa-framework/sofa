#ifndef __SELECTORVOL_H__
#define __SELECTORVOL_H__

#include "Container/fakeAttribute.h"
#include "Algo/DecimationVolumes/operator.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

enum SelectorType
{
	S_MapOrder,
	S_Random,
	S_EdgeLength,
	S_SG98,
	S_QEM
} ;

template <typename PFP> class ApproximatorGen ;
template <typename PFP, typename T, unsigned int ORBIT> class Approximator ;

/********************************************************************************
 *				 				Parent Selector									*
 ********************************************************************************/

//class du Selector de base
template <typename PFP>
class Selector
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL;

protected:

	MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;
	std::vector<ApproximatorGen<PFP>*>& m_approximators ;

public:
	Selector(MAP& m, VertexAttribute<typename PFP::VEC3>& pos, std::vector<ApproximatorGen<PFP>*>& approx) :
		m_map(m), m_position(pos), m_approximators(approx)
	{}
	virtual ~Selector()
	{}
	virtual SelectorType getType() = 0 ;
	virtual bool init() = 0 ;
	virtual bool nextEdge(Dart& d) = 0 ;
	virtual void updateBeforeCollapse(Dart d) = 0 ;
	virtual void updateAfterCollapse(Dart d2, Dart dd2) = 0 ;

	virtual void updateWithoutCollapse() = 0;
};

} // namespace Decimation

} // namespace Volume

} // namespace Algo

} // namespace CGoGN


#endif
