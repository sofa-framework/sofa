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
#ifndef CLOSESTPOINTREGISTRATIONFORCEFIELD_INL
#define CLOSESTPOINTREGISTRATIONFORCEFIELD_INL

#include "ClosestPointRegistrationForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <SofaLoader/MeshObjLoader.h>
#include <SofaGeneralEngine/NormalsFromPoints.h>
#include <limits>
#include <set>
#include <iterator>
#include <sofa/helper/gl/Color.h>

using std::cerr;
using std::endl;



namespace sofa
{

namespace component
{

namespace forcefield
{


using namespace helper;

template <class DataTypes>
ClosestPointRegistrationForceField<DataTypes>::ClosestPointRegistrationForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , ks(initData(&ks,(Real)100.0,"stiffness","uniform stiffness for the all springs."))
    , kd(initData(&kd,(Real)0.0,"damping","uniform damping for the all springs."))
    , cacheSize(initData(&cacheSize,(unsigned int)5,"cacheSize","number of closest points used in the cache to speed up closest point computation."))
    , blendingFactor(initData(&blendingFactor,(Real)0,"blendingFactor","blending between projection (=0) and attraction (=1) forces."))
    , outlierThreshold(initData(&outlierThreshold,(Real)0,"outlierThreshold","suppress outliers when distance > (meandistance + threshold*stddev)."))
    , normalThreshold(initData(&normalThreshold,(Real)0.5,"normalThreshold","suppress outliers when normal.closestPointNormal < threshold."))
    , projectToPlane(initData(&projectToPlane,true,"projectToPlane","project closest points in the plane defined by the normal."))
    , rejectBorders(initData(&rejectBorders,true,"rejectBorders","ignore border vertices."))
    , rejectOutsideBbox(initData(&rejectOutsideBbox,false,"rejectOutsideBbox","ignore source points outside bounding box of target points."))
    , sourceTriangles(initData(&sourceTriangles,"sourceTriangles","Triangles of the source mesh."))
    , sourceNormals(initData(&sourceNormals,"sourceNormals","Normals of the source mesh."))
    , targetPositions(initData(&targetPositions,"position","Vertices of the target mesh."))
    , targetNormals(initData(&targetNormals,"normals","Normals of the target mesh."))
    , targetTriangles(initData(&targetTriangles,"triangles","Triangles of the target mesh."))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis."))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow."))
    , drawColorMap(initData(&drawColorMap,false,"drawColorMap","Hue mapping of distances to closest point"))
    , theCloserTheStiffer(initData(&theCloserTheStiffer,false,"theCloserTheStiffer","Modify stiffness according to distance"))
{
}

template <class DataTypes>
ClosestPointRegistrationForceField<DataTypes>::~ClosestPointRegistrationForceField()
{
}


template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::reinit()
{

}

template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::init()
{
    this->Inherit::init();
    core::objectmodel::BaseContext* context = this->getContext();

    if(!(this->mstate)) this->mstate = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(context->getMechanicalState());

    // Get source triangles
    if(!sourceTriangles.getValue().size()) {
        sofa::component::loader::MeshObjLoader *meshobjLoader;
        this->getContext()->get( meshobjLoader, core::objectmodel::BaseContext::Local);
        if (meshobjLoader) {sourceTriangles.virtualSetLink(meshobjLoader->d_triangles); sout<<"imported triangles from "<<meshobjLoader->getName()<<sendl;}
    }
    // Get source normals
    if(!sourceNormals.getValue().size()) serr<<"normals of the source model not found"<<sendl;
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::detectBorder(vector<bool> &border,const helper::vector< tri > &triangles)
{
    unsigned int nbp=border.size();
    unsigned int nbt=triangles.size();
    for(unsigned int i=0;i<nbp;i++) border[i]=false;

    if(!nbt) return;
    vector<vector< unsigned int> > ngbTriangles((int)nbp);
    for(unsigned int i=0;i<nbt;i++) for(unsigned int j=0;j<3;j++)	ngbTriangles[triangles[i][j]].push_back(i);
    for(unsigned int i=0;i<nbp;i++) if(ngbTriangles[i].size()==0) border[i]=true;
    for(unsigned int i=0;i<nbt;i++)
        for(unsigned int j=0;j<3;j++)
        {
            unsigned int id1=triangles[i][j],id2=triangles[i][(j==2)?0:j+1];
            if(!border[id1] || !border[id2]) {
                bool bd=true;
                for(unsigned int i1=0;i1<ngbTriangles[id1].size() && bd;i1++)
                    for(unsigned int i2=0;i2<ngbTriangles[id2].size() && bd;i2++)
                        if(ngbTriangles[id1][i1]!=i)
                            if(ngbTriangles[id1][i1]==ngbTriangles[id2][i2])
                                bd=false;
                if(bd) border[id1]=border[id2]=true;
            }
        }
}


template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::initSource()
{
    // build k-d tree
    const VecCoord&  p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    if(p.size()) sourceKdTree.build(p);

    // detect border
    if(sourceBorder.size()!=p.size()) { sourceBorder.resize(p.size()); detectBorder(sourceBorder,sourceTriangles.getValue()); }
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::initTarget()
{
    // build k-d tree
    const VecCoord&  p = targetPositions.getValue();
    if(p.size()) targetKdTree.build(p);

    // updatebbox
    for(unsigned int i=0;i<p.size();++i)    targetBbox.include(p[i]);

    // detect border
    if(targetBorder.size()!=p.size()) { targetBorder.resize(p.size()); detectBorder(targetBorder,targetTriangles.getValue()); }
}



template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::updateClosestPoints()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord&  tp = targetPositions.getValue();

    unsigned int nbs=x.size(),nbt=tp.size();

    distanceSet emptyset;
    if(nbs!=closestSource.size()) {initSource();  closestSource.resize(nbs);	closestSource.fill(emptyset); cacheThresh_max.resize(nbs); cacheThresh_min.resize(nbs); previousX.assign(x.begin(),x.end());}
    if(nbt!=closestTarget.size()) {initTarget();  closestTarget.resize(nbt);	closestTarget.fill(emptyset);}

    this->sourceIgnored.resize(nbs); sourceIgnored.fill(false);
    this->targetIgnored.resize(nbt); targetIgnored.fill(false);

    if(nbs==0 || nbt==0) return;

    // closest target points from source points
    if(blendingFactor.getValue()<1) {

        //unsigned int count=0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0;i<(int)nbs;i++)
            if(rejectOutsideBbox.getValue() && !targetBbox.contains(x[i])) sourceIgnored[i]=true;
            else targetKdTree.getNClosestCached(closestSource[i], cacheThresh_max[i], cacheThresh_min[i], this->previousX[i], x[i], this->targetPositions.getValue(), this->cacheSize.getValue());
    }
    // closest source points from target points
    if(blendingFactor.getValue()>0)
    {
        initSource();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0;i<(int)nbt;i++)
            sourceKdTree.getNClosest(closestTarget[i],tp[i], this->mstate->read(core::ConstVecCoordId::position())->getValue(),1);
    }


    // prune outliers
    //    if(rejectOutsideBbox.getValue()) {
    //        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(!targetBbox.contains(x[closestTarget[i].begin()->second])) targetIgnored[i]=true;
    //    }
    if(outlierThreshold.getValue()!=0) {
        Real mean=0,stdev=0,count=0;
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) {count++; Real d=closestSource[i].begin()->first; stdev+=d*d; mean+=d; }
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) {count++; Real d=closestTarget[i].begin()->first; stdev+=d*d; mean+=d; }
        mean=mean/count; stdev=(Real)sqrt(stdev/count-mean*mean);
        mean+=stdev*outlierThreshold.getValue();
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(closestSource[i].begin()->first>mean) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(closestTarget[i].begin()->first>mean) targetIgnored[i]=true;
    }
    if(rejectBorders.getValue()) {
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(targetBorder[closestSource[i].begin()->second]) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(sourceBorder[closestTarget[i].begin()->second]) targetIgnored[i]=true;
    }
    if(normalThreshold.getValue()>(Real)-1. && sourceNormals.getValue().size()!=0 && targetNormals.getValue().size()!=0) {
        ReadAccessor< Data< VecCoord > > sn(sourceNormals);
        ReadAccessor< Data< VecCoord > > tn(targetNormals);
        for(unsigned int i=0;i<nbs;i++) if(closestSource[i].size()) if(dot(sn[i],tn[closestSource[i].begin()->second])<normalThreshold.getValue()) sourceIgnored[i]=true;
        for(unsigned int i=0;i<nbt;i++) if(closestTarget[i].size()) if(dot(tn[i],sn[closestTarget[i].begin()->second])<normalThreshold.getValue()) targetIgnored[i]=true;
    }
}




template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/,DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v )
{
    if(ks.getValue()==0) return;

    VecDeriv&        f = *_f.beginEdit();           //WDataRefVecDeriv f(_f);
    const VecCoord&  x = _x.getValue();			//RDataRefVecCoord x(_x);
    const VecDeriv&  v = _v.getValue();			//RDataRefVecDeriv v(_v);
    ReadAccessor< Data< VecCoord > > tn(targetNormals);
    ReadAccessor< Data< VecCoord > > tp(targetPositions);

    unsigned int nb = x.size();

    this->closestPos.resize(nb);

    updateClosestPoints();

    m_potentialEnergy = 0;

    // get attraction/ projection factors
    Real attrF=(Real) blendingFactor.getValue();
    if(attrF<(Real)0.) attrF=(Real)0.;
    if(attrF>(Real)1.) attrF=(Real)1.;
    Real projF=((Real)1.-attrF);

    if(tp.size()==0)
        for (unsigned int i=0; i<nb; i++)
            closestPos[i]=x[i];
    else
    {
        // count number of attractors
        cnt.resize(nb); cnt.fill(0);  if(attrF>0) for (unsigned int i=0; i<tp.size(); i++) if(!targetIgnored[i]) cnt[closestTarget[i].begin()->second]++;

        if(theCloserTheStiffer.getValue())
        {
            // find the min and the max distance value from source point to target point
            min=0;
            max=0;
            for (unsigned int i=0; i<x.size(); i++)
            {
                if(min==0 || min>closestSource[i].begin()->first) min=closestSource[i].begin()->first;
                if(max==0 || max<closestSource[i].begin()->first) max=closestSource[i].begin()->first;
            }
        }

        // compute targetpos = projF*closestto + attrF* sum closestfrom / count

        // projection to point or plane
        if(projF>0)
        {
            for (unsigned int i=0; i<nb; i++)
                if(!sourceIgnored[i])
                {
                    unsigned int id=closestSource[i].begin()->second;
                    if(projectToPlane.getValue() && tn.size()!=0)	closestPos[i]=(x[i]+tn[id]*dot(tp[id]-x[i],tn[id]))*projF;
                    else closestPos[i]=tp[id]*projF;
                    if(!cnt[i]) closestPos[i]+=x[i]*attrF;
                }
                else
                {
                    closestPos[i]=x[i]*projF;
                    if(!cnt[i]) closestPos[i]+=x[i]*attrF;
                }
        }
        else for (unsigned int i=0; i<nb; i++) { if(!cnt[i]) closestPos[i]=x[i]; else closestPos[i].fill(0); }

        // attraction
        if(attrF>0)
            for (unsigned int i=0; i<tp.size(); i++)
                if(!targetIgnored[i])
                {
                    unsigned int id=closestTarget[i].begin()->second;
                    closestPos[id]+=tp[i]*attrF/(Real)cnt[id];
                    sourceIgnored[id]=false;
                }
    }


    // add rest spring force
    for (unsigned int i=0; i<nb; i++)
    {
        //serr<<"addForce() between "<<i<<" and "<<closestPos[i]<<sendl;
        Coord u = this->closestPos[i]-x[i];
        Real nrm2 = u.norm2();
        Real k = this->ks.getValue();
        if(theCloserTheStiffer.getValue())
        {
            Real elongation = helper::rsqrt(nrm2);
            Real ks_max=k;
            Real ks_min=k*0.1;
            k = ks_min*(max-elongation)/(max-min)+ks_max*(elongation-min)/(max-min);
        }
        f[i]+=k*u;
        m_potentialEnergy += nrm2 * k * 0.5;
        if(this->kd.getValue() && nrm2) f[i]-=this->kd.getValue()*u*dot(u,v[i])/nrm2;
    }
    _f.endEdit();

}





template <class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams,DataVecDeriv& _df , const DataVecDeriv&  _dx )
{
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()) * this->ks.getValue();
    if(!k) return;
    sofa::helper::WriteAccessor< DataVecDeriv > df = _df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx = _dx;
    for (unsigned int i=0; i<dx.size(); i++)        df[i] -= dx[i] * k;
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams,const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()) * this->ks.getValue();
    if(!k) return;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix *mat = mref.matrix;
    const int offset = (int)mref.offset;
    const int N = Coord::total_size;
    const int nb = this->closestPos.size();
    for (int index = 0; index < nb; index++)
        for(int i = 0; i < N; i++)
            mat->add(offset + N * index + i, offset + N * index + i, -k);
}

template<class DataTypes>
void ClosestPointRegistrationForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if(ks.getValue()==0) return;

    if (!vparams->displayFlags().getShowForceFields() && !drawColorMap.getValue()) return;

    ReadAccessor< Data< VecCoord > > x(*this->getMState()->read(core::ConstVecCoordId::position()));
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    unsigned int nb = this->closestPos.size();
    if (vparams->displayFlags().getShowForceFields())
    {
        std::vector< defaulttype::Vector3 > points;
        for (unsigned int i=0; i<nb; i++)
            if(!sourceIgnored[i])
            {
                defaulttype::Vector3 point1 = DataTypes::getCPos(x[i]);
                defaulttype::Vector3 point2 = DataTypes::getCPos(this->closestPos[i]);
                points.push_back(point1);
                points.push_back(point2);
            }

        const defaulttype::Vec<4,float> c(0,1,0.5,1);
        if (showArrowSize.getValue()==0 || drawMode.getValue() == 0)	vparams->drawTool()->drawLines(points, 1, c);
        else if (drawMode.getValue() == 1)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawCylinder(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else if (drawMode.getValue() == 2)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawArrow(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else serr << "No proper drawing mode found!" << sendl;
    }

    if(drawColorMap.getValue())
    {
        std::vector< Real > dists(x.size());  for (unsigned int i=0; i<dists.size(); i++) dists[i]=0.;
        for (unsigned int i=0; i<nb; i++)
            if(!sourceIgnored[i])
            {
                defaulttype::Vector3 point1 = DataTypes::getCPos(x[i]);
                defaulttype::Vector3 point2 = DataTypes::getCPos(this->closestPos[i]);
                dists[i]=(point2-point1).norm();
            }
        Real max=0; for (unsigned int i=0; i<dists.size(); i++) if(max<dists[i]) max=dists[i];

        glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
        glDisable( GL_LIGHTING);

        ReadAccessor< Data< helper::vector< tri > > > t(sourceTriangles);
        if(t.size()) // mesh visu
        {
            glBegin( GL_TRIANGLES);
            for ( unsigned int i = 0; i < t.size(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    const unsigned int& indexP = t[i][j];
                    sofa::helper::gl::Color::setHSVA(dists[indexP]*240./max,1.,.8,1.);
                    glVertex3d(x[indexP][0],x[indexP][1],x[indexP][2]);
                }
            }
            glEnd();
        }

        else // point visu
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for (unsigned int i=0; i<dists.size(); i++)
            {
                sofa::helper::gl::Color::setHSVA(dists[i]*240./max,1.,.8,1.);
                glVertex3d(x[i][0],x[i][1],x[i][2]);
            }
            glEnd();
            glPointSize( 1);
        }

        glPopAttrib();
    }
#endif
}


}
}
} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_ClosestPointRegistrationForceField_INL */
