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
#ifndef SOFA_COMPONENT_ENGINE_INERTIAALIGN_H
#define SOFA_COMPONENT_ENGINE_INERTIAALIGN_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/component/component.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/defaulttype/Vec.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
namespace sofa
{

namespace component
{


class InertiaAlign: public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(InertiaAlign,core::objectmodel::BaseObject);

    InertiaAlign();
    ~InertiaAlign();
    typedef defaulttype::Mat<3,3> Mat3x3;

    /**
      * Data Fields
      */
    /// input
    Data <sofa::defaulttype::Vector3> targetC;
    Data <sofa::defaulttype::Vector3> sourceC;

    Data < Mat3x3 > targetInertiaMatrix;

    Data < Mat3x3 > sourceInertiaMatrix;
    /// input//output
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > m_positiont;
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > m_positions;
    helper::vector<sofa::defaulttype::Vec<3,SReal> > positionDistSource;

    /// Initialization method called at graph modification, during bottom-up traversal.
    virtual void init();

protected:

    typedef defaulttype::Vector3 Vector3;
    typedef defaulttype::Matrix4 Matrix4;

    SReal computeDistances(helper::vector<sofa::defaulttype::Vec<3,SReal> >, helper::vector<sofa::defaulttype::Vec<3,SReal> >);
    SReal distance(sofa::defaulttype::Vec<3,SReal>, helper::vector<sofa::defaulttype::Vec<3,SReal> >);
    SReal abs(SReal);

    Matrix4 inverseTransform(Matrix4);
    /**
      * Protected methods
      */
public:


};


} // namespace component

} // namespace sofa

#endif
