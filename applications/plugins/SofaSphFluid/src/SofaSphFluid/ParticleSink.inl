/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_PARTICLESINK_INL
#define SOFA_COMPONENT_MISC_PARTICLESINK_INL

#include <SofaSphFluid/config.h>
#include <SofaSphFluid/ParticleSink.h>

namespace sofa
{

namespace component
{

namespace misc
{


template<class DataTypes>
ParticleSink<DataTypes>::ParticleSink()
    : planeNormal(initData(&planeNormal, "normal", "plane normal"))
    , planeD0(initData(&planeD0, (Real)0, "d0", "plane d coef at which particles acceleration is constrained to 0"))
    , planeD1(initData(&planeD1, (Real)0, "d1", "plane d coef at which particles are removed"))
    , color(initData(&color, defaulttype::RGBAColor(0.0f,0.5f,0.2f,1.0f), "color", "plane color. (default=[0.0,0.5,0.2,1.0])"))
    , showPlane(initData(&showPlane, false, "showPlane", "enable/disable drawing of plane"))
    , fixed(initData(&fixed, "fixed", "indices of fixed particles"))
{
    this->f_listening.setValue(true);
    Deriv n;
    DataTypes::set(n, 0, 1, 0);
    planeNormal.setValue(n);
}

template<class DataTypes>
ParticleSink<DataTypes>::~ParticleSink()
{

}


template<class DataTypes>
void ParticleSink<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    if (!this->mstate) return;

    msg_info() << "Normal=" << planeNormal.getValue() << " d0=" << planeD0.getValue() << " d1=" << planeD1.getValue();

    sofa::core::topology::BaseMeshTopology* _topology;
    _topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters for topology data and handler
    fixed.createTopologicalEngine(_topology);
    fixed.registerTopologicalData();
}


template<class DataTypes>
void ParticleSink<DataTypes>::animateBegin(double /*dt*/, double time)
{
    if (!this->mstate) 
        return;
    
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    int n = x.size();
    helper::vector<unsigned int> remove;
    for (int i=n-1; i>=0; --i) // always remove points in reverse order
    {
        Real d = x[i]*planeNormal.getValue()-planeD1.getValue();
        if (d<0)
        {
            msg_info() << "SINK particle "<<i<<" time "<<time<<" position "<<x[i]<<" velocity "<<v[i] ;
            remove.push_back(i);
        }
    }
    if (!remove.empty())
    {
        sofa::component::topology::PointSetTopologyModifier* pointMod;
        this->getContext()->get(pointMod);

        if (pointMod != nullptr)
        {
            msg_info() << "Remove: " << remove.size() << " out of: " << n <<" particles using PointSetTopologyModifier.";
            pointMod->removePointsWarning(remove);
            pointMod->propagateTopologicalChanges();
            pointMod->removePointsProcess(remove);
        }
        else if(container::MechanicalObject<DataTypes>* object = dynamic_cast<container::MechanicalObject<DataTypes>*>(this->mstate.get()))
        {
            msg_info() << "Remove "<<remove.size()<<" particles using MechanicalObject.";
            // deleting the vertices
            for (unsigned int i = 0; i < remove.size(); ++i)
            {
                --n;
                object->replaceValue(n, remove[i] );
            }
            // resizing the state vectors
            this->mstate->resize(n);
        }
        else
        {
            msg_error() << "No external object supporting removing points!";
        }
    }
}

template<class DataTypes>
void ParticleSink<DataTypes>::projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dx)
{
    // Fixed points are directly done in projectVelocity
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(dx);
}

template<class DataTypes>
void ParticleSink<DataTypes>::projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  v )
{
    if (!this->mstate) return;

    VecDeriv& vel = *v.beginEdit(mparams);
    helper::ReadAccessor< Data<SetIndexArray> > _fixed = this->fixed;
    Deriv v0 = Deriv();
    for (unsigned int s = 0; s<_fixed.size(); s++)
    {
        vel[_fixed[s]] = v0;
    }    
    v.endEdit(mparams);
}


template<class DataTypes>
void ParticleSink<DataTypes>::projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData)
{
    if (!this->mstate) return;

    VecCoord& x = *xData.beginEdit(mparams);

    helper::WriteAccessor< Data< SetIndexArray > > _fixed = fixed;

    _fixed.clear();
    // constraint the last value
    for (unsigned int i=0; i<x.size(); i++)
    {
        Real d = x[i]*planeNormal.getValue()-planeD0.getValue();
        if (d<0)
        {
            _fixed.push_back(i);
        }
    }

    xData.endEdit(mparams);
}


template<class DataTypes>
void ParticleSink<DataTypes>::projectJacobianMatrix(const sofa::core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(cData);
}


template<class DataTypes>
void ParticleSink<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if(simulation::AnimateBeginEvent::checkEventType(event) )
    {
        simulation::AnimateBeginEvent* ev = static_cast<simulation::AnimateBeginEvent*>(event);
        animateBegin(ev->getDt(), this->getContext()->getTime());
    }
}

template<class DataTypes>
void ParticleSink<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!showPlane.getValue())
        return;

    vparams->drawTool()->saveLastState();

    defaulttype::Vec3d normal; normal = planeNormal.getValue();

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();
    const float size=1.0f;
    defaulttype::Vec3d center = normal*planeD0.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    vparams->drawTool()->disableLighting();
    vparams->drawTool()->setPolygonMode(0, true);

    sofa::defaulttype::RGBAColor _color(color.getValue()[0],color.getValue()[1],color.getValue()[2],1.0);
    std::vector<sofa::defaulttype::Vector3> vertices;

    vertices.push_back(sofa::defaulttype::Vector3(corners[0]));
    vertices.push_back(sofa::defaulttype::Vector3(corners[1]));
    vertices.push_back(sofa::defaulttype::Vector3(corners[2]));
    vertices.push_back(sofa::defaulttype::Vector3(corners[3]));
    vparams->drawTool()->drawQuad(vertices[0],vertices[1],vertices[2],vertices[3], cross((vertices[1] - vertices[0]), (vertices[2] - vertices[0])), _color);

    vparams->drawTool()->restoreLastState();
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_PARTICLESINK_INL

