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
#include "CudaCollisionDetection.h"

namespace sofa
{

namespace gpu
{

namespace cuda
{

void CudaCollisionDetection::beginNarrowPhase()
{
    Inherit::beginNarrowPhase();
}

void CudaCollisionDetection::addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    Inherit::addCollisionPair( cmPair );
}

void CudaCollisionDetection::endNarrowPhase()
{
    Inherit::endNarrowPhase();
}

/// Fill the info to send to the graphics cars
void CudaCollisionDetection::RigidRigidTest::init(GPUTest& test)
{
    const CudaVector<Vec3f>& p1 = elem1.getGrid()->meshPts;
    CudaDistanceGrid& g2 = *elem2.getGrid();
    test.nbPoints = p1.size();
    test.points = p1.deviceRead();
    test.gridnx = g2.getNx();
    test.gridny = g2.getNy();
    test.gridnz = g2.getNz();
    test.gridbbmin = g2.getBBMin();
    test.gridbbmax = g2.getBBMax();
    test.gridp0 = g2.getPMin();
    test.gridinvdp = g2.getInvCellWidth();
    test.grid = g2.getDists().deviceRead();
    test.margin = 0;
}

/// Create the list of SOFA contacts from the contacts detected by the GPU
void CudaCollisionDetection::RigidRigidTest::fillContacts(DetectionOutputVector& contacts)
{
}

/// Fill the info to send to the graphics cars
void CudaCollisionDetection::SphereRigidTest::init(GPUTest& test)
{
    const CudaVector<Vec3f>& p1 = *elem1.getCollisionModel()->getMechanicalState()->getX();
    CudaDistanceGrid& g2 = *elem2.getGrid();
    test.nbPoints = p1.size();
    test.points = p1.deviceRead();
    test.gridnx = g2.getNx();
    test.gridny = g2.getNy();
    test.gridnz = g2.getNz();
    test.gridbbmin = g2.getBBMin();
    test.gridbbmax = g2.getBBMax();
    test.gridp0 = g2.getPMin();
    test.gridinvdp = g2.getInvCellWidth();
    test.grid = g2.getDists().deviceRead();
    test.margin = elem1.r();
}

/// Create the list of SOFA contacts from the contacts detected by the GPU
void CudaCollisionDetection::SphereRigidTest::fillContacts(DetectionOutputVector& contacts)
{
}


} // namespace cuda

} // namespace gpu

} // namespace sofa
