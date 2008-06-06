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
//
// C++ Models: EdgeSetController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_INL
#define SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_INL

#include <sofa/component/controller/EdgeSetController.h>
#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <sofa/component/topology/ManifoldEdgeSetTopology.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
EdgeSetController<DataTypes>::EdgeSetController()
    :step(0.0)
{

}



template <class DataTypes>
void EdgeSetController<DataTypes>::init()
{
    edges = dynamic_cast<topology::EdgeSetTopology<DataTypes> * > (this->getContext()->getMainTopology());

    if (!edges)
        std::cerr << "WARNING. EdgeSetController has no binding Edges Topology\n";

    Inherit::init();

    if (edges)
    {
        edge0RestedLength = edges->getEdgeSetGeometryAlgorithms()->computeRestEdgeLength(0);
        edges->getEdgeSetTopologyContainer()->getEdgeVertexShellArray();
    }
}



template <class DataTypes>
void EdgeSetController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
    switch (mev->getState())
    {
    case sofa::core::objectmodel::MouseEvent::Wheel :
        this->mouseMode = Inherit::Wheel;
        step = 2 * (Real)(abs(mev->getWheelDelta()) / mev->getWheelDelta());
        break;

    default:
        break;
    }
}

template <class DataTypes>
void EdgeSetController<DataTypes>::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kev)
{
    switch(kev->getKey())
    {
    case '+':
        this->mouseMode = Inherit::Wheel;
        step = 4;
        break;
    case '-':
        this->mouseMode = Inherit::Wheel;
        step = -4;
        break;
    }
}

template <class DataTypes>
void EdgeSetController<DataTypes>::onBeginAnimationStep()
{
    applyController();
}



template <class DataTypes>
void EdgeSetController<DataTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    if (this->mouseMode == Inherit::Wheel)
    {
        this->mouseMode = Inherit::None;

        if (this->mState)
        {
            sofa::helper::Quater<Real>& quatrot = (*this->mState->getX0())[0].getOrientation();
            sofa::defaulttype::Vec<3,Real> vectrans(step * this->mainDirection[0] * (Real)0.05, step * this->mainDirection[1] * (Real)0.05, step * this->mainDirection[2] * (Real)0.05);
            vectrans = quatrot.rotate(vectrans);
            //	std::cout << "X0 += " << vectrans << std::endl;
            (*this->mState->getX0())[0].getCenter() += vectrans;
            //	(*this->mState->getX())[0].getCenter() += vectrans;
        }

        sofa::simulation::tree::GNode *node = static_cast<sofa::simulation::tree::GNode*> (this->getContext());
        sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
        sofa::simulation::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);

        modifyTopology();
    }
}



template <class DataTypes>
void EdgeSetController<DataTypes>::modifyTopology(void)
{
    assert(edges != 0);

    if (step >= 0)
    {
        edges->getEdgeSetTopologyContainer()->getEdgeVertexShellArray();

        sofa::helper::vector< unsigned int > baseEdge(0);
        baseEdge = edges->getEdgeSetTopologyContainer()->getEdgeVertexShell(0);

        if (baseEdge.size() == 1)
        {
            if (edges->getEdgeSetGeometryAlgorithms()->computeRestEdgeLength(baseEdge[0]) > ( 2 * edge0RestedLength ))
            {
                // First Edge makes 2
                sofa::helper::vector<unsigned int> indices(0);
                indices.push_back(baseEdge[0]);

                edges->getEdgeSetTopologyAlgorithms()->splitEdges(indices);

                // Renumber Vertices

                unsigned int numPoints = edges->getPointSetTopologyContainer()->getNumberOfVertices();

                sofa::helper::vector<unsigned int> permutations(numPoints);
                permutations[0] = 0;
                permutations[numPoints - 1] = 1;

                for (unsigned int i = 1; i < numPoints - 1; i++)
                    permutations[i] = i + 1;

                /*
                std::cout << "permutations : ";
                for (unsigned int i = 0; i < numPoints; i++)
                	std::cout << permutations[i] << "  ";
                std::cout << std::endl;
                */

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for (unsigned int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                /*
                std::cout << "inverse_permutations : ";
                for (unsigned int i = 0; i < numPoints; i++)
                	std::cout << inverse_permutations[i] << "  ";
                std::cout << std::endl;
                */

                edges->getEdgeSetTopologyAlgorithms()->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);

            }
        }
    }
    else
    {
        edges->getEdgeSetTopologyContainer()->getEdgeVertexShellArray();

        sofa::helper::vector< unsigned int > baseEdge;
        baseEdge = edges->getEdgeSetTopologyContainer()->getEdgeVertexShell(1);

        if (baseEdge.size() == 2)
        {

            if ((edges->getEdgeSetGeometryAlgorithms()->computeRestEdgeLength(baseEdge[0]) < ( 0.5 * edge0RestedLength ))
                ||(edges->getEdgeSetGeometryAlgorithms()->computeRestEdgeLength(baseEdge[1]) < ( 0.5 * edge0RestedLength )))

            {
                // Fuse Edges (0-1)

                sofa::helper::vector< sofa::helper::vector<unsigned int> > edges_fuse(0);
                sofa::helper::vector<unsigned int> v(0);
                v.push_back(baseEdge[0]);
                v.push_back(baseEdge[1]);
                edges_fuse.push_back(v);
                edges->getEdgeSetTopologyAlgorithms()->fuseEdges(edges_fuse, true);

                // Renumber Vertices

                unsigned int numPoints = edges->getPointSetTopologyContainer()->getNumberOfVertices();

                sofa::helper::vector<unsigned int> permutations(numPoints);
                permutations[0] = 0;
                permutations[1] = numPoints - 1;
                for (unsigned int i = 2; i < numPoints; i++)
                    permutations[i] = i-1;

                /*
                std::cout << "permutations : ";
                for (unsigned int i = 0; i < numPoints; i++)
                	std::cout << permutations[i] << "  ";
                std::cout << std::endl;
                */

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for (unsigned int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                /*
                std::cout << "inverse_permutations : ";
                for (unsigned int i = 0; i < numPoints; i++)
                	std::cout << inverse_permutations[i] << "  ";
                std::cout << std::endl;
                */

                edges->getEdgeSetTopologyAlgorithms()->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);

            }
        }
    }
}


template <class DataTypes>
void EdgeSetController<DataTypes>::draw()
{
    glDisable(GL_LIGHTING);

    if (edges)
    {
        glBegin(GL_LINES);
        for (unsigned int i=0; i<edges->getEdgeSetTopologyContainer()->getNumberOfEdges(); i++)
        {
            glColor4f(1.0,1.0,0.0,1.0);
            helper::gl::glVertexT((*this->mState->getX())[edges->getEdge(i)[0]]);
            helper::gl::glVertexT((*this->mState->getX())[edges->getEdge(i)[1]]);
        }
        glEnd();

        glPointSize(10);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<edges->getEdgeSetTopologyContainer()->getNumberOfEdges(); i++)
        {
            glColor4f(1.0,0.0,0.0,1.0);
            helper::gl::glVertexT((*this->mState->getX())[edges->getEdge(i)[0]]);
            helper::gl::glVertexT((*this->mState->getX())[edges->getEdge(i)[1]]);
        }
        glEnd();
        glPointSize(1);
    }
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_INL
