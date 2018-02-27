/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// C++ Interface: ParticleSink
//
// Description:
//
//
// Author: Jeremie Allard, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_MISC_PARTICLESINK_H
#define SOFA_COMPONENT_MISC_PARTICLESINK_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <ostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace misc
{

template<class TDataTypes>
class ParticleSink : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticleSink,TDataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef helper::vector<Real> VecDensity;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalModel;
    typedef helper::vector<unsigned int> SetIndexArray;

    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

    Data<Deriv> planeNormal; ///< plane normal
    Data<Real> planeD0; ///< plane d coef at which particles acceleration is constrained to 0
    Data<Real> planeD1; ///< plane d coef at which particles are removed
    Data<defaulttype::RGBAColor> color; ///< plane color. (default=[0.0,0.5,0.2,1.0])
    Data<bool> showPlane; ///< enable/disable drawing of plane

    sofa::component::topology::PointSubsetData< SetIndexArray > fixed; ///< indices of fixed particles
    //Data< SetIndexArray > fixed; ///< indices of fixed particles
protected:
    ParticleSink()
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

    virtual ~ParticleSink()
    {
    }
public:
    virtual void init() override
    {
        this->core::behavior::ProjectiveConstraintSet<TDataTypes>::init();
        if (!this->mstate) return;

        sout << "ParticleSink: normal="<<planeNormal.getValue()<<" d0="<<planeD0.getValue()<<" d1="<<planeD1.getValue()<<sendl;

        sofa::core::topology::BaseMeshTopology* _topology;
        _topology = this->getContext()->getMeshTopology();

        // Initialize functions and parameters for topology data and handler
        fixed.createTopologicalEngine(_topology);
        fixed.registerTopologicalData();

    }

    virtual void animateBegin(double /*dt*/, double time)
    {
        //sout << "ParticleSink: animate begin time="<<time<<sendl;
        if (!this->mstate) return;
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
            //sofa::core::topology::BaseMeshTopology* _topology;
            //_topology = this->getContext()->getMeshTopology();

            sofa::component::topology::PointSetTopologyModifier* pointMod;
            this->getContext()->get(pointMod);

            if (pointMod != NULL)
            {
                msg_info() << "ParticleSink: remove "<<remove.size()<<" particles using PointSetTopologyModifier.";
                pointMod->removePointsWarning(remove);
                pointMod->propagateTopologicalChanges();
                pointMod->removePointsProcess(remove);
            }
            else if(container::MechanicalObject<DataTypes>* object = dynamic_cast<container::MechanicalObject<DataTypes>*>(this->mstate.get()))
            {
                msg_info() << "ParticleSink: remove "<<remove.size()<<" particles using MechanicalObject.";
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
                msg_info() << "ERROR(ParticleSink): no external object supporting removing points!";
            }
        }
    }


    template <class DataDeriv>
    void projectResponseT(DataDeriv& res) ///< project dx to constrained space
    {
        if (!this->mstate) return;
        if (fixed.getValue().empty()) return;

        const SetIndexArray& _fixed = fixed.getValue();
        // constraint the last value
        for (unsigned int s=0; s<_fixed.size(); s++)
            res[_fixed[s]] = Deriv();
    }

    virtual void projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dx) override ///< project dx to constrained space
    {
        VecDeriv& res = *dx.beginEdit(mparams);
        projectResponseT(res);
        dx.endEdit(mparams);
    }

    virtual void projectVelocity(const sofa::core::MechanicalParams* /* mparams */, DataVecDeriv& /* v */) override ///< project dx to constrained space (dx models a velocity) override
    {

    }

    virtual void projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData) override ///< project x to constrained space (x models a position) override
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

    virtual void projectJacobianMatrix(const sofa::core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {

    }

    virtual void animateEnd(double /*dt*/, double /*time*/)
    {

    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override
    {
        if(simulation::AnimateBeginEvent::checkEventType(event) )
        {
            simulation::AnimateBeginEvent* ev = static_cast<simulation::AnimateBeginEvent*>(event);
            animateBegin(ev->getDt(), this->getContext()->getTime());
        }
        else if(simulation::AnimateEndEvent::checkEventType(event) )
        {
            simulation::AnimateEndEvent* ev = static_cast<simulation::AnimateEndEvent*>(event);
            animateEnd(ev->getDt(), this->getContext()->getTime());
        }
    }

    virtual void draw(const core::visual::VisualParams*) override
    {
#ifndef SOFA_NO_OPENGL
        if (!showPlane.getValue()) return;
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

        // glEnable(GL_LIGHTING);
        glDisable(GL_LIGHTING);
        glEnable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glCullFace(GL_FRONT);

        glColor3f(color.getValue()[0],color.getValue()[1],color.getValue()[2]);

        glBegin(GL_QUADS);
        helper::gl::glVertexT(corners[0]);
        helper::gl::glVertexT(corners[1]);
        helper::gl::glVertexT(corners[2]);
        helper::gl::glVertexT(corners[3]);
        glEnd();

        glDisable(GL_CULL_FACE);

        glColor4f(1,0,0,1);
#endif /* SOFA_NO_OPENGL */
    }
};

}

} // namespace component

} // namespace sofa

#endif

