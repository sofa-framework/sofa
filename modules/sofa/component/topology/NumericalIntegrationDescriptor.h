/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_H
#define SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_H

#include <sofa/defaulttype/Vec.h>
#include <map>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

/// Cross product for 3-elements vectors.
template< typename class Real, int N>
class NumericalIntegrationDescriptor {
public:
	typedef typename Vec<N, Real> BarycentricCoordinatesType;
	typedef std::pair<BarycentricCoordinatesType,Real> QuadraturePoint;
	typedef sofa::helper::vector<QuadraturePoint> QuadraturePointArray;
	
	typedef enum {
		GAUSS_METHOD =0,
		GAUSS_LOBATO_METHOD=1
	} QuadratureMethod; 
	typedef size_t IntegrationOrder;
	typedef std::pair<QuadratureMethod,IntegrationOrder> QuadratureMethodKey;

protected:
	std::map<typename QuadratureMethodKey,typename QuadraturePointArray>  quadratureMap;
public:
	/// empty constructor
	NumericalIntegrationDescriptor(){}
	/// returns the set of quadrature points associated with a given quadrature method and integration order
	QuadraturePointArray getQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order) const;
	/// returns all the indices corresponding to all available quadrature methods
	sofa::helper::set<QuadratureMethod> getQuadratureMethods() const;
	/// returns the quadrature integration orders available for a given method
	sofa::helper::set<IntegrationOrder> getIntegrationOrders(const QuadratureMethod qt) const;
	/// add a quadrature method in the map
	void addQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order, QuadraturePointArray qpa);
};


} // namespace topology

} // namespace component

} // namespace sofa

#endif
