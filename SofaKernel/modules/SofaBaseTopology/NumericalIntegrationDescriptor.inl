/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_INL
#define SOFA_COMPONENT_TOPOLOGY_NUMERICALINTEGRATIONDESCRIPTOR_INL

#include <SofaBaseTopology/NumericalIntegrationDescriptor.h>
#include <map>

namespace sofa
{

namespace component
{

namespace topology
{

template< typename Real, int N>
typename NumericalIntegrationDescriptor<Real,N>::QuadraturePointArray NumericalIntegrationDescriptor<Real,N>::getQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order) const
{
	QuadratureMethodKey key(qt,order);
	typename std::map<QuadratureMethodKey, QuadraturePointArray>::const_iterator it=quadratureMap.find(key);
	if (it!=quadratureMap.end())
		return ((*it).second);
	else {
		QuadraturePointArray qpa;
		return(qpa);
	}

}
template< typename Real, int N>
std::set<typename NumericalIntegrationDescriptor<Real,N>::QuadratureMethod>  NumericalIntegrationDescriptor<Real,N>::getQuadratureMethods() const
{
    std::set<QuadratureMethod>  qmset;
	typename std::map<QuadratureMethodKey, QuadraturePointArray>::const_iterator it;
	for (it=quadratureMap.begin();it!=quadratureMap.end();it++) {
		qmset.insert((*it).first.first);
	}
	return(qmset);
}
template< typename Real, int N>
std::set<typename NumericalIntegrationDescriptor<Real,N>::IntegrationOrder>  NumericalIntegrationDescriptor<Real,N>::getIntegrationOrders(const QuadratureMethod qt) const
{
    std::set<IntegrationOrder>  ioset;
	typename std::map<QuadratureMethodKey, QuadraturePointArray>::const_iterator it;
	for (it=quadratureMap.begin();it!=quadratureMap.end();it++) {
		if (((*it).first.first)==qt){
			ioset.insert((*it).first.second);
		}
	}
	return(ioset);
}
template< typename Real, int N>
void NumericalIntegrationDescriptor<Real,N>::addQuadratureMethod(const QuadratureMethod qt, const IntegrationOrder order, QuadraturePointArray qpa)
{
	quadratureMap.insert(std::pair<QuadratureMethodKey,QuadraturePointArray>(QuadratureMethodKey(qt,order),qpa));
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif
