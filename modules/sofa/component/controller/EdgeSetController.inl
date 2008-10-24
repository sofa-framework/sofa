/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
EdgeSetController<DataTypes>::EdgeSetController()
    : step(initData(&step,(Real)0.1,"step","base step when changing beam length"))
    , minLength(initData(&minLength,(Real)1.0,"minLength","min beam length"))
    , maxLength(initData(&maxLength,(Real)200.0,"maxLength","max beam length"))
    , maxDepl(initData(&maxDepl,(Real)0.5,"maxDepl","max depl when changing beam length"))
    , speed(initData(&speed,(Real)0.0,"speed","continuous beam length increase/decrease"))
    , reversed(initData(&reversed,false,"reversed","Extend or retract edgeSet from end"))
    , depl(0.0)
{

}



template <class DataTypes>
void EdgeSetController<DataTypes>::init()
{
    _topology = this->getContext()->getMeshTopology();
    this->getContext()->get(edgeGeo);
    this->getContext()->get(edgeMod);

    if (edgeGeo == NULL)
        logWarning("EdgeSetController has no binding EdgeSetGeometryAlgorithms");

    if (edgeMod == NULL)
        logWarning("EdgeSetController has no binding EdgeSetTopologyModifier");

    Inherit::init();

    /*
    if (_topology->getNbEdges()>0)
    {
    	edge0RestedLength = edgeGeo->computeRestEdgeLength(0);
    }
    */

    computeVertexT();

    if (!reversed.getValue())
    {
        if (vertexT.size() >= 2)
            edgeTLength = fabs(vertexT[1]-vertexT[0]);
        else
            edgeTLength = 1;
    }
    else
    {
        int n = this->mState->getSize();
        if (n > 1)
        {
            if (vertexT.size() >= 2)
                edgeTLength = fabs(vertexT[n-2] - vertexT[n-1]);
            else
                edgeTLength = 1;
        }
    }
}



template <class DataTypes>
void EdgeSetController<DataTypes>::computeVertexT()
{
    int n = this->mState->getSize();
    const VecCoord& x0 = * this->mState->getX0();

    vertexT.resize(n);

    if (!reversed.getValue())
    {
        for (int i = 0; i < n; ++i)
        {
            if (i == 0)
                vertexT[0] = 0;
            else
                vertexT[i] = vertexT[i-1] + ((x0[i] - x0[i-1]).norm());
        }

        if (n > 0)
            refPos = x0[0];
    }
    else
    {
        if (n > 0)
        {
            for (int i = n-1; i >= 0; i--)
            {
                if (i == n-1)
                    vertexT[i] = 0;
                else
                    vertexT[i] = vertexT[i+1] + ((x0[i+1] - x0[i]).norm());
            }

            refPos = x0[n-1];
        }
    }
}



template <class DataTypes>
void EdgeSetController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
    switch (mev->getState())
    {
    case sofa::core::objectmodel::MouseEvent::Wheel :
        this->mouseMode = Inherit::Wheel;
        if (mev->getWheelDelta() != 0.0)
            depl += step.getValue() * (Real)(abs(mev->getWheelDelta()) / mev->getWheelDelta());
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
        depl += 2*step.getValue();
        break;

    case '-':
        this->mouseMode = Inherit::Wheel;
        depl -= 2*step.getValue();
        break;
    case '*':
        this->mouseMode = Inherit::Wheel;
        speed.setValue(speed.getValue() + 10*step.getValue());
        break;

    case '/':
        this->mouseMode = Inherit::Wheel;
        speed.setValue(speed.getValue() - 10*step.getValue());
        break;
    case '0':
        this->mouseMode = Inherit::None;
        speed.setValue(0);
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

    if (depl != 0 || speed.getValue() != 0) //this->mouseMode == Inherit::Wheel)
    {
        depl += speed.getValue() * this->getContext()->getDt();
        this->mouseMode = Inherit::None;

        if (this->mState)
        {
            if (!reversed.getValue())
            {
                /*
                Coord& pos = (*this->mState->getX0())[0];
                pos = getNewRestPos(pos, vertexT[0], depl);
                vertexT[0] -= depl;
                depl = 0;
                */

                Coord& pos = (*this->mState->getX0())[0];
                double d;

                if (maxDepl.getValue() == 0 || fabs(depl) < maxDepl.getValue())
                {
                    d = depl;
                    depl = 0;
                }
                else
                {
                    d = (depl < 0) ? -maxDepl.getValue() : maxDepl.getValue();
                    depl -= d;
                }

                double endT = vertexT[vertexT.size()-1];
                double newT = vertexT[0];
                double sign = (endT > newT) ? 1.0 : -1.0;
                newT -= d;
                //std::cout << "length = " << sign*(endT-newT) << std::endl;

                if (sign*(endT-newT) > maxLength.getValue())
                {
                    //std::cout << "max length" << std::endl;
                    newT = endT - sign*maxLength.getValue();
                    d = vertexT[0] - newT;
                }
                else if (sign*(endT-newT) < minLength.getValue())
                {
                    //std::cout << "min length" << std::endl;
                    newT = endT - sign*minLength.getValue();
                    d = vertexT[0] - newT;
                }

                if (newT != vertexT[0])
                {
                    pos = getNewRestPos(pos, vertexT[0], d);
                    vertexT[0] = newT;
                }
                else
                    return;

            }
            else
            {
                /*
                int n = this->mState->getSize();
                if (n > 0)
                {
                	Coord& pos = (*this->mState->getX0())[n-1];
                	pos = getNewRestPos(pos, vertexT[n-1], depl);
                	vertexT[n-1] -= depl;
                	depl = 0;
                }
                */

                int n = this->mState->getSize();
                if (n > 0)
                {
                    Coord& pos = (*this->mState->getX0())[n-1];
                    double d;

                    if (maxDepl.getValue() == 0 || fabs(depl) < maxDepl.getValue())
                    {
                        d = depl;
                        depl = 0;
                    }
                    else
                    {
                        d = (depl < 0) ? -maxDepl.getValue() : maxDepl.getValue();
                        depl -= d;
                    }

                    double endT = vertexT[0];
                    double newT = vertexT[n-1];
                    double sign = (endT > newT) ? 1.0 : -1.0;

                    newT -= d;

                    if (sign*(endT-newT) > maxLength.getValue())
                    {
                        newT = endT - sign*maxLength.getValue();
                        d = vertexT[n-1] - newT;
                    }
                    else if (sign*(endT-newT) < minLength.getValue())
                    {
                        newT = endT - sign*minLength.getValue();
                        d = vertexT[n-1] - newT;
                    }

                    if (newT != vertexT[n-1])
                    {
                        pos = getNewRestPos(pos, vertexT[n-1], d);
                        vertexT[n-1] = newT;
                    }
                    else
                        return;
                }
            }
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
    assert(edgeGeo != 0);

    // Split

    if (!reversed.getValue())
    {
        sofa::helper::vector< unsigned int > baseEdge(0);
        baseEdge = _topology->getEdgeVertexShell(0);

        if (baseEdge.size() == 1)
        {
            if (fabs(vertexT[1] - vertexT[0]) > ( 2 * edgeTLength ))
            {
                // First Edge makes 2
                sofa::helper::vector<unsigned int> indices(0);
                indices.push_back(baseEdge[0]);

                edgeMod->splitEdges(indices);

                // Update vertexT
                vertexT.insert(vertexT.begin()+1, (vertexT[0] + vertexT[1])/static_cast<Real>(2.0));

                // Renumber vertices
                unsigned int numPoints = _topology->getNbPoints();

                // Last created vertex must come on the second position of the position vector.
                sofa::helper::vector<unsigned int> permutations(numPoints);
                permutations[0] = 0;
                permutations[numPoints - 1] = 1;
                for (unsigned int i = 1; i < numPoints - 1; i++)
                    permutations[i] = i + 1;

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for (unsigned int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);
            }
        }
    }
    else
    {
        int n = this->mState->getSize();

        if (n > 1)
        {
            sofa::helper::vector< unsigned int > baseEdge(0);
            baseEdge = _topology->getEdgeVertexShell(n - 1);

            if (baseEdge.size() == 1)
            {
                if (fabs(vertexT[n-2] - vertexT[n-1]) > ( 2 * edgeTLength ))
                {
                    // First Edge makes 2
                    sofa::helper::vector<unsigned int> indices(0);
                    indices.push_back(baseEdge[0]);

                    edgeMod->splitEdges(indices);

                    // update vertexT
                    vertexT.insert(vertexT.end() - 1, (vertexT[n-1] + vertexT[n-2])/static_cast<Real>(2.0));

                    // Renumber Vertices
                    unsigned int numPoints = _topology->getNbPoints();

                    // Last created vertex must come on the last but one position of the position vector.
                    sofa::helper::vector<unsigned int> permutations(numPoints);
                    for (unsigned int i = 0; i < numPoints - 2; i++)
                        permutations[i] = i;
                    permutations[numPoints - 2] = numPoints - 1;
                    permutations[numPoints - 1] = numPoints - 2;

                    sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                    for (unsigned int i = 0; i < numPoints; i++)
                        inverse_permutations[permutations[i]] = i;

                    edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) permutations, (const sofa::helper::vector<unsigned int> &) inverse_permutations);
                }
            }
        }
    }

    // Fuse

    if (!reversed.getValue())
    {
        sofa::helper::vector< unsigned int > baseEdge;
        baseEdge = _topology->getEdgeVertexShell(1);

        if (baseEdge.size() == 2)
        {
            if (fabs(vertexT[1] - vertexT[0]) < ( 0.5 * edgeTLength ))
            {
                // Fuse Edges (0-1)
                sofa::helper::vector< sofa::helper::vector<unsigned int> > edges_fuse(0);
                sofa::helper::vector<unsigned int> v(0);
                v.push_back(baseEdge[0]);
                v.push_back(baseEdge[1]);
                edges_fuse.push_back(v);
                edgeMod->fuseEdges(edges_fuse, true);

                // update vertexT
                vertexT.erase(vertexT.begin()+1);

                // Renumber Vertices
                unsigned int numPoints = _topology->getNbPoints();

                // The vertex to delete has to be set to the last position of the position vector.
                sofa::helper::vector<unsigned int> permutations(numPoints);
                permutations[0] = 0;
                permutations[1] = numPoints - 1;
                for (unsigned int i = 2; i < numPoints; i++)
                    permutations[i] = i-1;

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for (unsigned int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);
            }
        }
    }
    else
    {
        int n = this->mState->getSize();

        if (n > 1)
        {
            sofa::helper::vector< unsigned int > baseEdge;
            baseEdge = _topology->getEdgeVertexShell(n - 2);

            if (baseEdge.size() == 2)
            {
                if (fabs(vertexT[n-2] - vertexT[n-1]) < ( 0.5 * edgeTLength ))
                {
                    // Fuse Edges (n-2, n-1)
                    sofa::helper::vector< sofa::helper::vector<unsigned int> > edges_fuse(0);
                    sofa::helper::vector< unsigned int > v(0);

                    v.push_back(baseEdge[0]);
                    v.push_back(baseEdge[1]);
                    edges_fuse.push_back(v);
                    edgeMod->fuseEdges(edges_fuse, true);

                    // update vertexT
                    vertexT.erase(vertexT.end() - 2);

                    // Renumber Vertices
                    unsigned int numPoints = _topology->getNbPoints();

                    // The vertex to delete has to be set to the last position of the position vector.
                    sofa::helper::vector<unsigned int> permutations(numPoints);
                    for (unsigned int i = 0; i < numPoints; i++)
                        permutations[i] = i;

                    sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                    for (unsigned int i = 0; i < numPoints; i++)
                        inverse_permutations[permutations[i]] = i;

                    edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) permutations, (const sofa::helper::vector<unsigned int> &) inverse_permutations);
                }
            }
        }
    }
}


template <class DataTypes>
void EdgeSetController<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;

    glDisable(GL_LIGHTING);

    if (edgeGeo)
    {
        glBegin(GL_LINES);
        for (int i=0; i<_topology->getNbEdges(); i++)
        {
            glColor4f(1.0,1.0,0.0,1.0);
            helper::gl::glVertexT((*this->mState->getX())[_topology->getEdge(i)[0]]);
            helper::gl::glVertexT((*this->mState->getX())[_topology->getEdge(i)[1]]);
        }
        glEnd();

        glPointSize(10);
        glBegin(GL_POINTS);
        for (int i=0; i<_topology->getNbEdges(); i++)
        {
            glColor4f(1.0,0.0,0.0,1.0);
            helper::gl::glVertexT((*this->mState->getX())[_topology->getEdge(i)[0]]);
            helper::gl::glVertexT((*this->mState->getX())[_topology->getEdge(i)[1]]);
        }
        glEnd();
        glPointSize(1);
    }
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_INL
