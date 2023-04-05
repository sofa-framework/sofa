/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/mapping/nonlinear/RigidRigidMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/io/XspLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>

#include <sofa/core/MechanicalParams.h>
#include <cstring>

#ifndef SOFA_BUILD_SOFA_COMPONENT_MAPPING_NONLINEAR
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#endif

// This component has been DEPRECATED since SOFA v23.06 and will be removed in SOFA v23.12.
// Please use RigidMapping with template='Rigid3,Rigid3' instead.
// If this component is crucial to you please report that to sofa-dev@ so we can reconsider this component for future re-integration.
namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
RigidRigidMapping<TIn,TOut>::RigidRigidMapping()
    : Inherit(),
      d_points(initData(&d_points, "initialPoints", "Initial position of the points")),
      d_repartition(initData(&d_repartition,"repartition","number of child frames per parent frame. \n"
                           "If empty, all the children are attached to the parent with index \n"
                           "given in the \"index\" attribute. If one value, each parent frame drives \n"
                           "the given number of children frames. Otherwise, the values are the number \n"
                           "of child frames driven by each parent frame. ")),
      d_index(initData(&d_index,sofa::Index(0),"index","input frame index")),
      d_fileRigidRigidMapping(initData(&d_fileRigidRigidMapping,"filename","Xsp file where to load rigidrigid mapping description")),
      d_axisLength(initData( &d_axisLength, 0.7, "axisLength", "axis length for display")),
      d_indexFromEnd( initData ( &d_indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") ),
      d_globalToLocalCoords ( initData ( &d_globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
{
    this->addAlias(&d_fileRigidRigidMapping,"fileRigidRigidMapping");
}


template <class TIn, class TOut>
class RigidRigidMapping<TIn, TOut>::Loader :
        public helper::io::XspLoaderDataHook,
        public helper::io::SphereLoaderDataHook
{
public:
    RigidRigidMapping<TIn, TOut>* dest;
    Loader(RigidRigidMapping<TIn, TOut>* dest) : dest(dest) {}

    void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal, SReal, SReal, bool, bool) override
    {
        OutCoord c;
        Out::set(c,px,py,pz);
        dest->d_points.beginEdit()->push_back(c);
        dest->d_points.endEdit();
    }

    void addSphere(SReal px, SReal py, SReal pz, SReal) override
    {
        OutCoord c;
        Out::set(c,px,py,pz);
        dest->d_points.beginEdit()->push_back(c);
        dest->d_points.endEdit();
    }
};

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::load(const char *filename)
{
    OutVecCoord& pts = *d_points.beginEdit();
    pts.resize(0);

    if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".xs3"))
    {
        Loader loader(this);
        sofa::helper::io::XspLoader::Load(filename, loader);
    }
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".sph"))
    {
        Loader loader(this);
        sofa::helper::io::SphereLoader::Load(filename, loader);
    }
    else if (strlen(filename)>0)
    {
        // Default to mesh loader
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh!=nullptr)
        {
            pts.resize(mesh->getVertices().size());
            for (sofa::Index i=0; i<mesh->getVertices().size(); i++)
            {
                Out::set(pts[i], mesh->getVertices()[i][0], mesh->getVertices()[i][1], mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }

    d_points.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::init()
{
    this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);

    if (!d_fileRigidRigidMapping.getValue().empty())
        this->load(d_fileRigidRigidMapping.getFullPath().c_str());

    if (d_points.getValue().empty() && this->toModel!=nullptr)
    {
        helper::ReadAccessor<Data<sofa::Index>> index = d_index;
        helper::ReadAccessor<Data<type::vector<sofa::Size>>> repartition = d_repartition;

        const OutVecCoord& xto =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        sofa::Size toModelSize = xto.size();
        const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
        sofa::Size fromModelSize = xfrom.size();

        helper::WriteAccessor<Data<OutVecCoord>> pts = d_points;
        pts.resize(toModelSize);
        sofa::Index i=0, cpt=0;

        if(d_globalToLocalCoords.getValue() == true)
        {
            switch (repartition.size())
            {
            case 0 :
                if (index >= fromModelSize) {
                    msg_error() << "Invalid index for mapping. Size of parent is " << fromModelSize << " and given index is " << index;
                    this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
                    return;
                }

                for (i=0; i<toModelSize; i++){
                    globalToLocalCoords(pts[i], xfrom[index], xto[i]);
                }

                break;
            case 1 :               
                for (i=0; i<fromModelSize; i++)
                {
                    for(sofa::Index j=0; j<repartition[0]; j++, cpt++)
                    {
                        if (cpt >= toModelSize) {
                            msg_error() << "Invalid repartition for mapping. Size of child is " << toModelSize;
                            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
                            return;
                        }
                        globalToLocalCoords(pts[cpt], xfrom[i], xto[cpt]);
                    }
                }
                break;
            default :
                if (repartition.size() != fromModelSize) {
                    msg_error() << "Invalid repartition for mapping. Size of parent is " << fromModelSize;
                    this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
                    return;
                }
                for (i=0; i<fromModelSize; i++)
                {
                    for(sofa::Index j=0; j<repartition[i]; j++,cpt++)
                    {
                        if (cpt >= toModelSize) {
                            msg_error() << "Invalid repartition for mapping. Size of child is " << toModelSize;
                            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
                            return;
                        }
                        globalToLocalCoords(pts[cpt], xfrom[i], xto[cpt]);
                    }
                }
                break;
            }
        }
        else
        {
            for (i=0; i<toModelSize; i++)
                pts[i] = xto[i];
        }
    }

    this->Inherit::init();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::globalToLocalCoords(OutCoord& result, const InCoord& xfrom, const OutCoord& x)
{
    result.getCenter() = xfrom.getOrientation().inverse().rotate( x.getCenter() - xfrom.getCenter() ) ;
    result.getOrientation() = xfrom.getOrientation().inverse() * x.getOrientation() ;
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::clear()
{
    (*this->d_points.beginEdit()).clear();
    this->d_points.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::setRepartition(sofa::Size value)
{
    type::vector<sofa::Size>& rep = *d_repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    d_repartition.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::setRepartition(type::vector<sofa::Size> values)
{
    type::vector<sofa::Size>& rep = *d_repartition.beginEdit();
    rep.clear();
    rep.reserve(values.size());
    auto it = values.begin();
    while (it != values.end())
    {
        rep.push_back(*it);
        it++;
    }
    d_repartition.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    helper::WriteAccessor< Data<OutVecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

    sofa::Index cptOut;
    sofa::Size val;

    out.resize(d_points.getValue().size());
    m_pointsR0.resize(d_points.getValue().size());

    switch (d_repartition.getValue().size())
    {
    case 0 : //no value specified : simple rigid mapping
        if (!d_indexFromEnd.getValue())
        {
            in[d_index.getValue()].writeRotationMatrix(m_rotation);
            for(sofa::Index i=0; i<d_points.getValue().size(); i++)
            {
                m_pointsR0[i].getCenter() = m_rotation*(d_points.getValue()[i]).getCenter();
                out[i] = in[d_index.getValue()].mult(d_points.getValue()[i]);
            }
        }
        else
        {
            in[in.size() - 1 - d_index.getValue()].writeRotationMatrix(m_rotation);
            for(sofa::Index i=0; i<d_points.getValue().size(); i++)
            {
                m_pointsR0[i].getCenter() = m_rotation*(d_points.getValue()[i]).getCenter();
                out[i] = in[in.size() - 1 - d_index.getValue()].mult(d_points.getValue()[i]);
            }
        }
        break;

    case 1 : //one value specified : uniform repartition.getValue() mapping on the input dofs
        val = d_repartition.getValue()[0];
        cptOut=0;

        for (sofa::Index ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(m_rotation);
            for(sofa::Index ito=0; ito<val; ito++)
            {
                m_pointsR0[cptOut].getCenter() = m_rotation*(d_points.getValue()[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(d_points.getValue()[cptOut]);
                cptOut++;
            }
        }
        break;

    default: //n values are specified : heterogen repartition.getValue() mapping on the input dofs
        cptOut=0;

        for (sofa::Index ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(m_rotation);
            for(sofa::Index ito=0; ito<d_repartition.getValue()[ifrom]; ito++)
            {
                m_pointsR0[cptOut].getCenter() = m_rotation*(d_points.getValue()[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(d_points.getValue()[cptOut]);
                cptOut++;
            }
        }
        break;
    }
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    helper::WriteAccessor< Data<OutVecDeriv> > childVelocities = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > parentVelocities = dIn;

    Vector v,omega;
    childVelocities.resize(d_points.getValue().size());
    sofa::Index cptchildVelocities;
    sofa::Size val;

    switch (d_repartition.getValue().size())
    {
    case 0:
        if (!d_indexFromEnd.getValue())
        {
            v = getVCenter(parentVelocities[d_index.getValue()]);
            omega = getVOrientation(parentVelocities[d_index.getValue()]);

            for( size_t i=0 ; i< childVelocities.size() ; ++i)
            {
                getVCenter(childVelocities[i]) =  v + cross(omega,m_pointsR0[i].getCenter());
                getVOrientation(childVelocities[i]) = omega;
            }
        }
        else
        {
            v = getVCenter(parentVelocities[parentVelocities.size() - 1 - d_index.getValue()]);
            omega = getVOrientation(parentVelocities[parentVelocities.size() - 1 - d_index.getValue()]);

            for( size_t i=0 ; i< childVelocities.size() ; ++i)
            {
                getVCenter(childVelocities[i]) =  v + cross(omega,m_pointsR0[i].getCenter());
                getVOrientation(childVelocities[i]) = omega;
            }
        }
        break;

    case 1:
        val = d_repartition.getValue()[0];
        cptchildVelocities=0;
        for (sofa::Index ifrom=0 ; ifrom<parentVelocities.size(); ifrom++)
        {
            v = getVCenter(parentVelocities[ifrom]);
            omega = getVOrientation(parentVelocities[ifrom]);

            for(sofa::Index ito=0; ito<val; ito++,cptchildVelocities++)
            {
                getVCenter(childVelocities[cptchildVelocities]) =  v + cross(omega,(m_pointsR0[cptchildVelocities]).getCenter());
                getVOrientation(childVelocities[cptchildVelocities]) = omega;
            }
        }
        break;

    default:
        cptchildVelocities=0;
        for (sofa::Index ifrom=0 ; ifrom<parentVelocities.size(); ifrom++)
        {
            v = getVCenter(parentVelocities[ifrom]);
            omega = getVOrientation(parentVelocities[ifrom]);

            for(sofa::Index ito=0; ito<d_repartition.getValue()[ifrom]; ito++,cptchildVelocities++)
            {
                getVCenter(childVelocities[cptchildVelocities]) =  v + cross(omega,(m_pointsR0[cptchildVelocities]).getCenter());
                getVOrientation(childVelocities[cptchildVelocities]) = omega;
            }
        }
        break;
    }

}


template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    helper::WriteAccessor< Data<InVecDeriv> > parentForces = dOut;
    helper::ReadAccessor< Data<OutVecDeriv> > childForces = dIn;

    Vector v,omega;
    sofa::Index childrenPerParent;
    sofa::Index childIndex = 0;
    sofa::Index parentIndex;

    switch(d_repartition.getValue().size())
    {
    case 0 :
        for( ; childIndex< childForces.size() ; ++childIndex)
        {
            // out = Jt in
            // Jt = [ I     ]
            //      [ -OM^t ]
            // -OM^t = OM^

            Vector f = getVCenter(childForces[childIndex]);
            v += f;
            omega += getVOrientation(childForces[childIndex]) + cross(f,-m_pointsR0[childIndex].getCenter());
        }

        parentIndex = d_indexFromEnd.getValue() ? sofa::Index(parentForces.size()-1-d_index.getValue()) : d_index.getValue();
        getVCenter(parentForces[parentIndex]) += v;
        getVOrientation(parentForces[parentIndex]) += omega;
        break;
    case 1 :
        childrenPerParent = d_repartition.getValue()[0];
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            v=Vector();
            omega=Vector();
            for(sofa::Index i=0; i<childrenPerParent; i++, childIndex++)
            {
                Vector f = getVCenter(childForces[childIndex]);
                v += f;
                omega += getVOrientation(childForces[childIndex]) + cross(f,-m_pointsR0[childIndex].getCenter());
            }
            getVCenter(parentForces[parentIndex]) += v;
            getVOrientation(parentForces[parentIndex]) += omega;
        }
        break;
    default :
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            v=Vector();
            omega=Vector();
            for(sofa::Index i=0; i<d_repartition.getValue()[parentIndex]; i++, childIndex++)
            {
                Vector f = getVCenter(childForces[childIndex]);
                v += f;
                omega += getVOrientation(childForces[childIndex]) + cross(f,-m_pointsR0[childIndex].getCenter());
            }
            getVCenter(parentForces[parentIndex]) += v;
            getVOrientation(parentForces[parentIndex]) += omega;
        }
        break;
    }

}



template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceChangeId, core::ConstMultiVecDerivId )
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    helper::ReadAccessor<Data<OutVecDeriv> > childForces (*mparams->readF(this->toModel.get()));
    helper::WriteAccessor<Data<InVecDeriv> > parentForces (*parentForceChangeId[this->fromModel.get()].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacements (*mparams->readDx(this->fromModel.get()));
    Real kfactor = (Real)mparams->kFactor();

    sofa::Index childrenPerParent;
    sofa::Index childIndex = 0;
    sofa::Index parentIndex;

    switch(d_repartition.getValue().size())
    {
    case 0 :
        parentIndex = d_indexFromEnd.getValue() ? sofa::Index(parentForces.size()-1-d_index.getValue()) : d_index.getValue();
        for( ; childIndex< childForces.size() ; ++childIndex)
        {
            typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
            const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
            parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, m_pointsR0[childIndex].getCenter()) * kfactor;
        }

        break;
    case 1 :
        childrenPerParent = d_repartition.getValue()[0];
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {            
            for( size_t i=0 ; i<childrenPerParent ; ++i, ++childIndex)
            {
                typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
                const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
                parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, m_pointsR0[childIndex].getCenter()) * kfactor;
            }
        }
        break;
    default :
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            for( size_t i=0 ; i<d_repartition.getValue()[parentIndex] ; i++, childIndex++)
            {
                typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
                const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
                parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, m_pointsR0[childIndex].getCenter()) * kfactor;
            }
        }
        break;
    }

}






template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<OutMatrixDeriv>& dIn)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    switch (d_repartition.getValue().size())
    {
    case 0:
    {
        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            Vector v, omega;

            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const OutDeriv data = colIt.val();
                // out = Jt in
                // Jt = [ I     ]
                //      [ -OM^t ]
                // -OM^t = OM^
                Vector f = getVCenter(data);
                v += f;
                omega += getVOrientation(data) + cross(f,-m_pointsR0[colIt.index()].getCenter());
            }

            const InDeriv result(v, omega);
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            if (!d_indexFromEnd.getValue())
            {
                o.addCol(d_index.getValue(), result);
            }
            else
            {
                // Commented by PJ. Bug??
                // o.addCol(out.size() - 1 - index.getValue(), result);
                const sofa::Size numDofs = this->getFromModel()->getSize();
                o.addCol(numDofs - 1 - d_index.getValue(), result);
            }
        }

        break;
    }
    case 1:
    {
        const sofa::Size numDofs = this->getFromModel()->getSize();
        const sofa::Size val = d_repartition.getValue()[0];

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            sofa::Index cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (sofa::Index ito = 0; ito < numDofs; ito++)
            {
                Vector v, omega;
                bool needToInsert = false;

                for (sofa::Index r = 0; r < val && colIt != colItEnd; r++, cpt++)
                {
                    if (sofa::Index(colIt.index()) != cpt)
                        continue;

                    needToInsert = true;
                    const OutDeriv data = colIt.val();
                    Vector f = getVCenter(data);
                    v += f;
                    omega += getVOrientation(data) + cross(f,-m_pointsR0[cpt].getCenter());

                    ++colIt;
                }

                if (needToInsert)
                {
                    const InDeriv result(v, omega);

                    typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                    o.addCol(ito, result);
                }
            }
        }

        break;
    }
    default:
    {
        const sofa::Size numDofs = this->getFromModel()->getSize();

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            sofa::Index cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (sofa::Index ito = 0; ito < numDofs; ito++)
            {
                Vector v, omega;
                bool needToInsert = false;

                for (sofa::Index r = 0; r < d_repartition.getValue()[ito] && colIt
                        != colItEnd; r++, cpt++)
                {
                    if (sofa::Index(colIt.index()) != cpt)
                        continue;

                    needToInsert = true;

                    const OutDeriv data = colIt.val();
                    const Vector f = getVCenter(data);
                    v += f;
                    omega += getVOrientation(data) + cross(f, -m_pointsR0[cpt].getCenter());

                    ++colIt;
                }

                if (needToInsert)
                {
                    const InDeriv result(v, omega);

                    typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                    o.addCol(ito, result);
                }
            }
        }

        break;
    }
    }

    dOut.endEdit();
}


template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::computeAccFromMapping(const core::MechanicalParams *mparams, Data<OutVecDeriv>& dAcc_out, const Data<InVecDeriv>& dV_in, const Data<InVecDeriv>& dAcc_in)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid)
        return;

    const InVecDeriv& v_in = dV_in.getValue();

    {
        OutVecDeriv& acc_out = *dAcc_out.beginEdit();
        acc_out.clear();
        acc_out.resize(d_points.getValue().size());
        dAcc_out.endEdit();
    }

    // current acceleration on acc_in is applied on the child (when more than one mapping)
    applyJ(mparams, dAcc_out, dAcc_in);

    OutVecDeriv& acc_out = *dAcc_out.beginEdit();

    // computation of the acceleration due to the current velocity
    // a+= w^(w^OM)

    Vector omega;
    sofa::Index cptchildV;
    sofa::Size val;

    switch (d_repartition.getValue().size())
    {
    case 0:

        if (!d_indexFromEnd.getValue())
        {
            omega = getVOrientation(v_in[d_index.getValue()]);
        }
        else
        {
            omega = getVOrientation(v_in[v_in.size() - 1 - d_index.getValue()]);
        }

        for(sofa::Index i=0; i<d_points.getValue().size(); i++)
        {
            getVCenter(acc_out[i]) +=   cross(omega, cross(omega,m_pointsR0[i].getCenter()) );
        }
        break;

    case 1:
        val = d_repartition.getValue()[0];
        cptchildV=0;
        for (sofa::Index ifrom=0 ; ifrom<v_in.size() ; ifrom++)
        {
            omega = getVOrientation(v_in[ifrom]);

            for(sofa::Index ito=0; ito<val; ito++)
            {
                getVCenter(acc_out[cptchildV]) +=  cross(omega, cross(omega,(m_pointsR0[cptchildV]).getCenter()) );
                cptchildV++;
            }
        }
        break;

    default:
        cptchildV=0;
        for (sofa::Index ifrom=0 ; ifrom<v_in.size() ; ifrom++)
        {
            omega = getVOrientation(v_in[ifrom]);

            for(sofa::Index ito=0; ito<d_repartition.getValue()[ifrom]; ito++)
            {
                getVCenter(acc_out[cptchildV]) += cross(omega, cross(omega,(m_pointsR0[cptchildV]).getCenter()) );
                cptchildV++;
            }
        }
        break;
    }

    dAcc_out.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (this->d_componentState.getValue() == core::objectmodel::ComponentState::Invalid){
        return;
    }

    if (!getShow(this,vparams)) {
        return;
    }

    const typename Out::VecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const type::Vec3& sizes = type::Vec3(d_axisLength.getValue(), d_axisLength.getValue(), d_axisLength.getValue());
    for (sofa::Index i=0; i<x.size(); i++)
    {
        vparams->drawTool()->drawFrame(x[i].getCenter(), x[i].getOrientation(), sizes);
    }
}

template <class TIn, class TOut>
bool RigidRigidMapping<TIn, TOut>::getShow(const core::objectmodel::BaseObject* /*m*/, const core::visual::VisualParams* vparams) const { return vparams->displayFlags().getShowMechanicalMappings(); }

template <class TIn, class TOut>
bool RigidRigidMapping<TIn, TOut>::getShow(const core::BaseMapping* /*m*/, const core::visual::VisualParams* vparams) const { return vparams->displayFlags().getShowMechanicalMappings(); }


} // namespace sofa::component::mapping::nonlinear
