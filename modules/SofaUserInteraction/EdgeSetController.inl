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

#include <SofaUserInteraction/EdgeSetController.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>

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
    , startingIndex(initData(&startingIndex,0,"startingIndex","index of the edge where a topological change occurs (negative for index relative to last point)"))
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
        serr << "EdgeSetController has no binding EdgeSetGeometryAlgorithms." << sendl;

    if (edgeMod == NULL)
        serr << "EdgeSetController has no binding EdgeSetTopologyModifier." << sendl;

    if (reversed.getValue() && startingIndex.getValue() != 0)
        serr << "WARNING : startingIndex different from 0 is not implemented for reversed case." << sendl;

    Inherit::init();

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
        /// useful here ??
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
    const VecCoord& x0 =  this->mState->read(core::ConstVecCoordId::restPosition())->getValue();

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

        /// Unused
        //	if (n > 0)
        //		refPos = x0[0];
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

            /// Unused
            //	refPos = x0[n-1];
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
void EdgeSetController<DataTypes>::onBeginAnimationStep(const double /*dt*/)
{
    applyController();
}



template <class DataTypes>
void EdgeSetController<DataTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    int startId = startingIndex.getValue();
    if (startId < 0) startId += this->mState->getSize();
    if (startId < 0) startId = 0;
    if (startId >= (int)this->mState->getSize()-1) startId = (int)this->mState->getSize()-1;

    if (depl != 0 || speed.getValue() != 0)
    {
        depl += (Real)(speed.getValue() * this->getContext()->getDt());
        this->mouseMode = Inherit::None;
        if (this->mState)
        {
            helper::WriteAccessor<Data<VecCoord> > x0 = *this->mState->write(core::VecCoordId::restPosition());
            if (!reversed.getValue())
            {
                if (startId != 0)
                {
                    for(int index_it = 0; index_it <= startId; index_it++)
                    {
                        Coord& pos = x0[index_it];

                        pos =  getNewRestPos(pos, vertexT[index_it], -depl);
                        vertexT[index_it] -= depl;
                    }
                    depl = 0;
                }
                else
                {
                    Coord& pos = x0[0];
                    Real d;

                    if (maxDepl.getValue() == 0 || fabs(depl) < maxDepl.getValue())
                    {
                        d = depl;
                        depl = 0;
                    }
                    else
                    {
                        d = (depl < 0) ? -maxDepl.getValue() : maxDepl.getValue();
                        depl -= d; //?
                    }

                    Real endT = vertexT[vertexT.size() - 1];
                    Real newT = vertexT[0];
                    Real sign = (endT > newT) ? 1.0f : -1.0f;
                    newT -= d;
                    //sout << "length = " << sign*(endT-newT) << sendl;

                    if (sign*(endT-newT) > maxLength.getValue())
                    {
                        //sout << "max length" << sendl;
                        newT = endT - sign*maxLength.getValue();   // Edge is too long : A new Edge will be introduced
                        d = vertexT[0] - newT;
                    }
                    else if (sign*(endT-newT) < minLength.getValue())
                    {
                        //sout << "min length" << sendl;
                        newT = endT - sign*minLength.getValue();   // Edge is too small : An Edge will be removed
                        d = vertexT[0] - newT;
                    }

                    if (newT != vertexT[0])							// if no displacement (d=0) then newT == vertexT[0]
                    {
                        pos = getNewRestPos(pos, vertexT[0], (Real)-d);
                        vertexT[0] = (Real)newT;
                    }
                    else
                        return;

                }



            }
            else // reversed  !
            {
                int n = this->mState->getSize();

                if (n > 0)
                {
                    Coord& pos = x0[n-1];
                    Real d;

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

                    Real endT = vertexT[0];
                    Real newT = vertexT[n-1];
                    Real sign = (endT > newT) ? 1.0f : -1.0f;

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
                        pos = getNewRestPos(pos, vertexT[n-1], (Real)-d);
                        vertexT[n-1] = (Real)newT;
                    }
                    else
                        return;

                }

            }
        }
        {
            sofa::simulation::Node *node = static_cast<sofa::simulation::Node*> (this->getContext());
            sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor(core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
            sofa::simulation::UpdateMappingVisitor updateVisitor(core::ExecParams::defaultInstance()); updateVisitor.execute(node);
        }
        if (modifyTopology())
        {
            sofa::simulation::Node *node = static_cast<sofa::simulation::Node*> (this->getContext());
            sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor(core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
            sofa::simulation::UpdateMappingVisitor updateVisitor(core::ExecParams::defaultInstance()); updateVisitor.execute(node);
        }
    }
    //serr<<"applyController ended"<<sendl;
}



template <class DataTypes>
bool EdgeSetController<DataTypes>::modifyTopology(void)
{
    assert(edgeGeo != 0);
    bool changed = false;
    int startId = startingIndex.getValue();
    if (startId < 0) startId += this->mState->getSize();
    if (startId < 0) startId = 0;
    if (startId >= (int)this->mState->getSize()-1) startId = (int)this->mState->getSize()-1;

    // Split

    if (!reversed.getValue())
    {
        sofa::helper::vector< unsigned int > baseEdge(0);
        baseEdge = _topology->getEdgesAroundVertex(startId);
        /// if the startingIndex is not on the base or on the tip of the wire, then we choose the second edge shared by the starting point.
        if (baseEdge.size() == 2)
        {
            //unsigned int startingEdge = baseEdge[1];
            //baseEdge.pop_back();
            //baseEdge[0] = startingEdge;
            // we keep the edge with the next point and not the previous point
            int e0 = baseEdge[0];
            int e1 = baseEdge[1];
            core::topology::BaseMeshTopology::Edge ei0 = _topology->getEdge(e0);
            core::topology::BaseMeshTopology::Edge ei1 = _topology->getEdge(e1);
            baseEdge.resize(1);
            if (ei0[0]+ei0[1] > ei1[0]+ei1[1])
                baseEdge[0] = e0;
            else
                baseEdge[0] = e1;
        }

        if (baseEdge.size() == 1)
        {
            if (fabs(vertexT[startId+1] - vertexT[startId]) > ( 2 * edgeTLength ))
            {
                helper::WriteAccessor<Data<VecCoord> > x0 = *this->mState->write(core::VecCoordId::restPosition());
                // First Edge makes 2
                sofa::helper::vector<unsigned int> indices(0);
                indices.push_back(baseEdge[0]);

                edgeMod->splitEdges(indices);
                // pos = pos of the last point of the wire
                Coord& pos = x0[this->mState->getSize() - 1];

                // point placed in the middle of the edge
                pos = getNewRestPos(	x0[startId],
                        vertexT[startId],
                        (vertexT[startId + 1] - vertexT[startId]) / static_cast<Real>(2.0));

                // Update vertexT
                vertexT.insert(	vertexT.begin()+ startId + 1,
                        (vertexT[startId] + vertexT[startId + 1]) / static_cast<Real>(2.0));

                // Renumber vertices
                int numPoints = _topology->getNbPoints();

                // Last created vertex must come on the startId + 1 position of the position vector.
                sofa::helper::vector<unsigned int> permutations(numPoints);
                for (int i = 0 ; i <= startId; i++)
                    permutations[i] = i;

                permutations[numPoints - 1] = startId+1;

                for ( int i = startId+1; i < numPoints - 1; i++)
                    permutations[i] = i + 1;

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for ( int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);
                changed = true;

                //startingIndex.setValue(startId+1);
            }
        }
    }
    else // reversed case // TODO implementation for startId !=0
    {
        int n = this->mState->getSize();

        if (n > 1)
        {
            sofa::helper::vector< unsigned int > baseEdge(0);
            baseEdge = _topology->getEdgesAroundVertex(n - 1);

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
                    changed = true;
                }
            }
        }
    }

    // Fuse

    if (!reversed.getValue())
    {
        sofa::helper::vector< unsigned int > baseEdge;
        baseEdge = _topology->getEdgesAroundVertex(startId+1);

        if (baseEdge.size()==1)
        {
            serr<<"Warning nothing to be removed"<<sendl;
        }

        if (baseEdge.size() == 2)
        {
            if (fabs(vertexT[startId+1] - vertexT[startId]) < ( 0.5 * edgeTLength ))
            {
                // Fuse Edges (0-1)
                sofa::helper::vector< sofa::helper::vector<unsigned int> > edges_fuse(0);
                sofa::helper::vector<unsigned int> v(0);
                v.push_back(baseEdge[0]);
                v.push_back(baseEdge[1]);
                edges_fuse.push_back(v);
                edgeMod->fuseEdges(edges_fuse, true);

                // update vertexT
                vertexT.erase(vertexT.begin() + startId + 1);

                // Renumber Vertices
                int numPoints = _topology->getNbPoints();

                // The vertex to delete has to be set to the last position of the position vector.
                sofa::helper::vector<unsigned int> permutations(numPoints);
                for ( int i = 0; i <= startId ; i++)
                {
                    permutations[i]=i;
                }

                permutations[startId+1] = numPoints - 1;

                for ( int i = startId+2; i < numPoints; i++)
                    permutations[i] = i-1;

                sofa::helper::vector<unsigned int> inverse_permutations(numPoints);
                for ( int i = 0; i < numPoints; i++)
                    inverse_permutations[permutations[i]] = i;

                edgeMod->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutations, (const sofa::helper::vector<unsigned int> &) permutations);
                changed = true;

                //if(startId > 0) startingIndex.setValue(startId-1);
            }
        }
    }
    else // reversed // TODO implementation for startId !=0
    {
        int n = this->mState->getSize();

        if (n > 1)
        {
            sofa::helper::vector< unsigned int > baseEdge;
            baseEdge = _topology->getEdgesAroundVertex(n - 2);

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
                    changed = true;
                }
            }
        }
    }
    return changed;
}


template <class DataTypes>
void EdgeSetController<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    glDisable(GL_LIGHTING);

    if (edgeGeo && this->mState)
    {
        helper::ReadAccessor<Data<VecCoord> > x = *this->mState->read(core::VecCoordId::position());

        glBegin(GL_LINES);
        for (int i=0; i<_topology->getNbEdges(); i++)
        {
            glColor4f(1.0,1.0,0.0,1.0);
            helper::gl::glVertexT(x[_topology->getEdge(i)[0]]);
            helper::gl::glVertexT(x[_topology->getEdge(i)[1]]);
        }
        glEnd();
        /*
        		glPointSize(10);
        		glBegin(GL_POINTS);
        		for (int i=0; i<_topology->getNbEdges(); i++)
        		{
        			glColor4f(1.0,0.0,0.0,1.0);
        			helper::gl::glVertexT((this->mstate->read(core::ConstVecCoordId::position())->getValue())[_topology->getEdge(i)[0]]);
        			helper::gl::glVertexT((this->mstate->read(core::ConstVecCoordId::position())->getValue())[_topology->getEdge(i)[1]]);
        		}
        		glEnd();
        		glPointSize(1);
        */
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_INL
