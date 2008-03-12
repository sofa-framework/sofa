/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
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
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_CYLINDERGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_CYLINDERGRIDTOPOLOGY_H

#include <sofa/component/topology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

class CylinderGridTopology : public GridTopology
{
public:
    typedef Vec3d Vec3;
    typedef double Real;

    CylinderGridTopology(int nx, int ny, int nz);
    CylinderGridTopology();

    unsigned getIndex( int i, int j, int k ) const; ///< one-dimensional index of a grid point
    Vec3 getPoint(int i) const;
    Vec3 getPoint(int x, int y, int z) const;
    bool hasPos()  const { return true; }
    double getPX(int i)  const { return getPoint(i)[0]; }
    double getPY(int i) const { return getPoint(i)[1]; }
    double getPZ(int i) const { return getPoint(i)[2]; }

protected:
    Data< Vec3 > center;
    Data< Vec3 > axis;
    Data< Real > radius, length;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
