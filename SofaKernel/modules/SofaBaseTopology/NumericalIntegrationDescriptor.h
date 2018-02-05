/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_H
#define SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_H
#include "config.h"

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>

namespace sofa
{

namespace component
{

namespace topology
{

/// Cross product for 3-elements vectors.
template< typename Real, int N>
class NumericalIntegrationDescriptor {
 
public:
    typedef sofa::defaulttype::Vec<N, Real> BarycentricCoordinatesType;
	typedef std::pair<BarycentricCoordinatesType,Real> QuadraturePoint;
	typedef sofa::helper::vector<QuadraturePoint> QuadraturePointArray;
	
	typedef enum {
		GAUSS_LEGENDRE_METHOD =0,
		GAUSS_LOBATO_METHOD=1,
		NEWTON_COTES_METHOD=2,
		GAUSS_SIMPLEX_METHOD=3,
		GAUSS_QUAD_METHOD=4,
		GAUSS_CUBE_METHOD=5
	} QuadratureMethod; 
	typedef size_t IntegrationOrder;
	typedef std::pair<QuadratureMethod,IntegrationOrder> QuadratureMethodKey;

protected:
	std::map<QuadratureMethodKey, QuadraturePointArray>  quadratureMap;
public:
	/// empty constructor
	NumericalIntegrationDescriptor(){}
	/// returns the set of quadrature points associated with a given quadrature method and integration order
	QuadraturePointArray getQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order) const;
	/// returns all the indices corresponding to all available quadrature methods
    std::set<QuadratureMethod> getQuadratureMethods() const;
	/// returns the quadrature integration orders available for a given method
    std::set<IntegrationOrder> getIntegrationOrders(const QuadratureMethod qt) const;
	/// add a quadrature method in the map
	void addQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order, QuadraturePointArray qpa);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<double, 4>;
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<double, 3>;
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<double, 1>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<float, 4>;
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<float, 3>;
extern template class SOFA_BASE_TOPOLOGY_API NumericalIntegrationDescriptor<float, 1>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
