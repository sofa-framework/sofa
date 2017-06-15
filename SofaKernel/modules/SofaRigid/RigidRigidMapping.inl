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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_INL

#include <SofaRigid/RigidRigidMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>

#include <string.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class RigidRigidMapping<TIn, TOut>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:
    RigidRigidMapping<TIn, TOut>* dest;
    Loader(RigidRigidMapping<TIn, TOut>* dest) : dest(dest) {}
    virtual void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal, SReal, SReal, bool, bool)
    {
        OutCoord c;
        Out::set(c,px,py,pz);
        dest->points.beginEdit()->push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
        dest->points.endEdit();
    }
    virtual void addSphere(SReal px, SReal py, SReal pz, SReal)
    {
        OutCoord c;
        Out::set(c,px,py,pz);
        dest->points.beginEdit()->push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
        dest->points.endEdit();
    }
};

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::load(const char *filename)
{
    OutVecCoord& pts = *points.beginEdit();
    pts.resize(0);

    if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".xs3"))
    {
        Loader loader(this);
        loader.helper::io::MassSpringLoader::load(filename);
    }
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".sph"))
    {
        Loader loader(this);
        loader.helper::io::SphereLoader::load(filename);
    }
    else if (strlen(filename)>0)
    {
        // Default to mesh loader
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh!=NULL)
        {
            pts.resize(mesh->getVertices().size());
            for (unsigned int i=0; i<mesh->getVertices().size(); i++)
            {
                Out::set(pts[i], mesh->getVertices()[i][0], mesh->getVertices()[i][1], mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }

    points.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::init()
{

    if (!fileRigidRigidMapping.getValue().empty())
        this->load(fileRigidRigidMapping.getFullPath().c_str());

    if (this->points.getValue().empty() && this->toModel!=NULL)
    {
        const OutVecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        OutVecCoord& pts = *points.beginEdit();

        pts.resize(x.size());
        unsigned int i=0, cpt=0;

        if(globalToLocalCoords.getValue() == true)
        {
            const typename In::VecCoord& xfrom =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
            switch (repartition.getValue().size())
            {
            case 0 :
                for (i = 0; i < x.size(); i++)
                {
                    // pts[i] = x[i] - xfrom[0];
                    pts[i].getCenter() = xfrom[index.getValue()].getOrientation().inverse().rotate( x[i].getCenter() - xfrom[index.getValue()].getCenter() ) ;
                    pts[i].getOrientation() = xfrom[index.getValue()].getOrientation().inverse() * x[i].getOrientation() ;
                }
                break;
            case 1 :
                for (i=0; i<xfrom.size(); i++)
                    for(unsigned int j=0; j<repartition.getValue()[0]; j++,cpt++)
                        pts[cpt] = x[cpt] - xfrom[i];
                break;
            default :
                for (i=0; i<xfrom.size(); i++)
                    for(unsigned int j=0; j<repartition.getValue()[i]; j++,cpt++)
                        pts[cpt] = x[cpt] - xfrom[i];
                break;
            }
        }
        else
        {
            for (i=0; i<x.size(); i++)
                pts[i] = x[i];
        }

        points.endEdit();
    }

    this->Inherit::init();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::clear()
{
    (*this->points.beginEdit()).clear();
    this->points.endEdit();
}

/*
template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::disable()
{
if (!this->points.getValue().empty() && this->toModel!=NULL)
{
VecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
x.resize(points.getValue().size());
for (unsigned int i=0;i<points.getValue().size();i++)
x[i] = points.getValue()[i];
}
}
*/

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::setRepartition(unsigned int value)
{
    helper::vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::setRepartition(sofa::helper::vector<unsigned int> values)
{
    helper::vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.reserve(values.size());
    //repartition.setValue(values);
    sofa::helper::vector<unsigned int>::iterator it = values.begin();
    while (it != values.end())
    {
        rep.push_back(*it);
        it++;
    }
    this->repartition.endEdit();
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

    unsigned int cptOut;
    unsigned int val;

    out.resize(points.getValue().size());
    pointsR0.resize(points.getValue().size());

    switch (repartition.getValue().size())
    {
    case 0 : //no value specified : simple rigid mapping
        if (!indexFromEnd.getValue())
        {
            in[index.getValue()].writeRotationMatrix(rotation);
            for(unsigned int i=0; i<points.getValue().size(); i++)
            {
                pointsR0[i].getCenter() = rotation*(points.getValue()[i]).getCenter();
                out[i] = in[index.getValue()].mult(points.getValue()[i]);
            }
        }
        else
        {
            in[in.size() - 1 - index.getValue()].writeRotationMatrix(rotation);
            for(unsigned int i=0; i<points.getValue().size(); i++)
            {
                pointsR0[i].getCenter() = rotation*(points.getValue()[i]).getCenter();
                out[i] = in[in.size() - 1 - index.getValue()].mult(points.getValue()[i]);
            }
        }
        break;

    case 1 : //one value specified : uniform repartition.getValue() mapping on the input dofs
        val = repartition.getValue()[0];
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(rotation);
            for(unsigned int ito=0; ito<val; ito++)
            {
                pointsR0[cptOut].getCenter() = rotation*(points.getValue()[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(points.getValue()[cptOut]);
                cptOut++;
            }
        }
        break;

    default: //n values are specified : heterogen repartition.getValue() mapping on the input dofs
        if (repartition.getValue().size() != in.size())
        {
            serr<<"Error : mapping dofs repartition.getValue() is not correct"<<sendl;
            return;
        }
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(rotation);
            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                pointsR0[cptOut].getCenter() = rotation*(points.getValue()[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(points.getValue()[cptOut]);
                cptOut++;
            }
        }
        break;
    }
}

template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    helper::WriteAccessor< Data<OutVecDeriv> > childVelocities = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > parentVelocities = dIn;

    Vector v,omega;
    childVelocities.resize(points.getValue().size());
    unsigned int cptchildVelocities;
    unsigned int val;

    switch (repartition.getValue().size())
    {
    case 0:
        if (!indexFromEnd.getValue())
        {
            v = getVCenter(parentVelocities[index.getValue()]);
            omega = getVOrientation(parentVelocities[index.getValue()]);

            for( size_t i=0 ; i<this->maskTo->size() ; ++i)
            {
                if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

                getVCenter(childVelocities[i]) =  v + cross(omega,pointsR0[i].getCenter());
                getVOrientation(childVelocities[i]) = omega;
            }
        }
        else
        {
            v = getVCenter(parentVelocities[parentVelocities.size() - 1 - index.getValue()]);
            omega = getVOrientation(parentVelocities[parentVelocities.size() - 1 - index.getValue()]);

            for( size_t i=0 ; i<this->maskTo->size() ; ++i)
            {
                if( !this->maskTo->getEntry(i) ) continue;

                getVCenter(childVelocities[i]) =  v + cross(omega,pointsR0[i].getCenter());
                getVOrientation(childVelocities[i]) = omega;
            }
        }
        break;

    case 1:
        val = repartition.getValue()[0];
        cptchildVelocities=0;
        for (unsigned int ifrom=0 ; ifrom<parentVelocities.size(); ifrom++)
        {
            v = getVCenter(parentVelocities[ifrom]);
            omega = getVOrientation(parentVelocities[ifrom]);

            for(unsigned int ito=0; ito<val; ito++,cptchildVelocities++)
            {
                if( this->maskTo->isActivated() && !this->maskTo->getEntry(cptchildVelocities) ) continue;

                getVCenter(childVelocities[cptchildVelocities]) =  v + cross(omega,(pointsR0[cptchildVelocities]).getCenter());
                getVOrientation(childVelocities[cptchildVelocities]) = omega;
            }
        }
        break;

    default:
        if (repartition.getValue().size() != parentVelocities.size())
        {
            serr<<"Error : mapping dofs repartition.getValue() is not correct"<<sendl;
            return;
        }
        cptchildVelocities=0;
        for (unsigned int ifrom=0 ; ifrom<parentVelocities.size(); ifrom++)
        {
            v = getVCenter(parentVelocities[ifrom]);
            omega = getVOrientation(parentVelocities[ifrom]);

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++,cptchildVelocities++)
            {
                if( this->maskTo->isActivated() && !this->maskTo->getEntry(cptchildVelocities) ) continue;

                getVCenter(childVelocities[cptchildVelocities]) =  v + cross(omega,(pointsR0[cptchildVelocities]).getCenter());
                getVOrientation(childVelocities[cptchildVelocities]) = omega;
            }
        }
        break;
    }

}


template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn)
{
    helper::WriteAccessor< Data<InVecDeriv> > parentForces = dOut;
    helper::ReadAccessor< Data<OutVecDeriv> > childForces = dIn;

    Vector v,omega;
    unsigned int childrenPerParent;
    unsigned childIndex = 0;
    unsigned parentIndex;

    ForceMask& mask = *this->maskFrom;


    switch(repartition.getValue().size())
    {
    case 0 :
        for( ; childIndex<this->maskTo->size() ; ++childIndex)
        {
            if( !this->maskTo->getEntry(childIndex) ) continue;

            // out = Jt in
            // Jt = [ I     ]
            //      [ -OM^t ]
            // -OM^t = OM^

            Vector f = getVCenter(childForces[childIndex]);
            v += f;
            omega += getVOrientation(childForces[childIndex]) + cross(f,-pointsR0[childIndex].getCenter());
        }

        parentIndex = indexFromEnd.getValue() ? parentForces.size()-1-index.getValue() : index.getValue();
        getVCenter(parentForces[parentIndex]) += v;
        getVOrientation(parentForces[parentIndex]) += omega;
        mask.insertEntry(parentIndex);

        //			if (!indexFromEnd.getValue())
        //			{
        //                            getVCenter(parentForces[index.getValue()]) += v;
        //                            getVOrientation(parentForces[index.getValue()]) += omega;
        //                            maskFrom->insertEntry(index.getValue());
        //			}
        //			else
        //			{
        //                            getVCenter(parentForces[parentForces.size() - 1 - index.getValue()]) += v;
        //                            getVOrientation(parentForces[parentForces.size() - 1 - index.getValue()]) += omega;
        //                            maskFrom->insertEntry(parentForces.size() - 1 - index.getValue());
        //			}

        break;
    case 1 :
        childrenPerParent = repartition.getValue()[0];
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            v=Vector();
            omega=Vector();
            for(unsigned int i=0; i<childrenPerParent; i++, childIndex++)
            {
                if( !this->maskTo->getEntry(childIndex) ) continue;

                Vector f = getVCenter(childForces[childIndex]);
                v += f;
                omega += getVOrientation(childForces[childIndex]) + cross(f,-pointsR0[childIndex].getCenter());
            }
            getVCenter(parentForces[parentIndex]) += v;
            getVOrientation(parentForces[parentIndex]) += omega;
            mask.insertEntry(parentIndex);
        }
        break;
    default :
        if (repartition.getValue().size() != parentForces.size())
        {
            serr<<"Error : mapping dofs repartition.getValue() is not correct"<<sendl;
            return;
        }
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            v=Vector();
            omega=Vector();
            for(unsigned int i=0; i<repartition.getValue()[parentIndex]; i++, childIndex++)
            {
                if( !this->maskTo->getEntry(childIndex) ) continue;

                Vector f = getVCenter(childForces[childIndex]);
                v += f;
                omega += getVOrientation(childForces[childIndex]) + cross(f,-pointsR0[childIndex].getCenter());
            }
            getVCenter(parentForces[parentIndex]) += v;
            getVOrientation(parentForces[parentIndex]) += omega;
            mask.insertEntry(parentIndex);
        }
        break;
    }

}



template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceChangeId, core::ConstMultiVecDerivId )
{
    helper::ReadAccessor<Data<OutVecDeriv> > childForces (*mparams->readF(this->toModel));
    helper::WriteAccessor<Data<InVecDeriv> > parentForces (*parentForceChangeId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacements (*mparams->readDx(this->fromModel));
    Real kfactor = (Real)mparams->kFactor();

    unsigned int childrenPerParent;
    unsigned childIndex = 0;
    unsigned parentIndex;

    switch(repartition.getValue().size())
    {
    case 0 :
        parentIndex = indexFromEnd.getValue() ? parentForces.size()-1-index.getValue() : index.getValue();
        for( ; childIndex<this->maskTo->size() ; ++childIndex)
        {
            if( !this->maskTo->getEntry(childIndex) ) continue;

            typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
            const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
            parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, pointsR0[childIndex].getCenter()) * kfactor;
        }

        break;
    case 1 :
        childrenPerParent = repartition.getValue()[0];
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {            
            for( size_t i=0 ; i<childrenPerParent ; ++i, ++childIndex)
            {
                if( !this->maskTo->getEntry(childIndex) ) continue;

                typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
                const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
                parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, pointsR0[childIndex].getCenter()) * kfactor;
            }
        }
        break;
    default :
        if (repartition.getValue().size() != parentForces.size())
        {
            serr<<"Error : mapping dofs repartition.getValue() is not correct"<<sendl;
            return;
        }
        for(parentIndex=0; parentIndex<parentForces.size(); parentIndex++)
        {
            for( size_t i=0 ; i<repartition.getValue()[parentIndex] ; i++, childIndex++)
            {
                if( !this->maskTo->getEntry(childIndex) ) continue;

                typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIndex]);
                const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIndex]);
                parentTorque -=  TIn::crosscross( getLinear(childForces[childIndex]), parentRotation, pointsR0[childIndex].getCenter()) * kfactor;
            }
        }
        break;
    }

}






template <class TIn, class TOut>
void RigidRigidMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<OutMatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    switch (repartition.getValue().size())
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
                omega += getVOrientation(data) + cross(f,-pointsR0[colIt.index()].getCenter());
            }

            const InDeriv result(v, omega);
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            if (!indexFromEnd.getValue())
            {
                o.addCol(index.getValue(), result);
            }
            else
            {
                // Commented by PJ. Bug??
                // o.addCol(out.size() - 1 - index.getValue(), result);
                const unsigned int numDofs = this->getFromModel()->getSize();
                o.addCol(numDofs - 1 - index.getValue(), result);
            }
        }

        break;
    }
    case 1:
    {
        const unsigned int numDofs = this->getFromModel()->getSize();
        const unsigned int val = repartition.getValue()[0];

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            unsigned int cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (unsigned int ito = 0; ito < numDofs; ito++)
            {
                Vector v, omega;
                bool needToInsert = false;

                for (unsigned int r = 0; r < val && colIt != colItEnd; r++, cpt++)
                {
                    if (colIt.index() != cpt)
                        continue;

                    needToInsert = true;
                    const OutDeriv data = colIt.val();
                    Vector f = getVCenter(data);
                    v += f;
                    omega += getVOrientation(data) + cross(f,-pointsR0[cpt].getCenter());

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
        const unsigned int numDofs = this->getFromModel()->getSize();

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            unsigned int cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (unsigned int ito = 0; ito < numDofs; ito++)
            {
                Vector v, omega;
                bool needToInsert = false;

                for (unsigned int r = 0; r < repartition.getValue()[ito] && colIt
                        != colItEnd; r++, cpt++)
                {
                    if (colIt.index() != cpt)
                        continue;

                    needToInsert = true;

                    const OutDeriv data = colIt.val();
                    const Vector f = getVCenter(data);
                    v += f;
                    omega += getVOrientation(data) + cross(f, -pointsR0[cpt].getCenter());

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
    const InVecDeriv& v_in = dV_in.getValue();
    //	const InVecDeriv& acc_in = dAcc_in.getValue();

    {
        OutVecDeriv& acc_out = *dAcc_out.beginEdit();
        acc_out.clear();
        acc_out.resize(points.getValue().size());
        dAcc_out.endEdit();
    }

    // current acceleration on acc_in is applied on the child (when more than one mapping)
    applyJ(mparams, dAcc_out, dAcc_in);

    OutVecDeriv& acc_out = *dAcc_out.beginEdit();

    // computation of the acceleration due to the current velocity
    // a+= w^(w^OM)

    Vector omega;
    unsigned int cptchildV;
    unsigned int val;

    switch (repartition.getValue().size())
    {
    case 0:

        if (!indexFromEnd.getValue())
        {
            omega = getVOrientation(v_in[index.getValue()]);
        }
        else
        {
            omega = getVOrientation(v_in[v_in.size() - 1 - index.getValue()]);
        }

        for(unsigned int i=0; i<points.getValue().size(); i++)
        {
            getVCenter(acc_out[i]) +=   cross(omega, cross(omega,pointsR0[i].getCenter()) );
        }
        break;

    case 1:
        val = repartition.getValue()[0];
        cptchildV=0;
        for (unsigned int ifrom=0 ; ifrom<v_in.size() ; ifrom++)
        {
            omega = getVOrientation(v_in[ifrom]);

            for(unsigned int ito=0; ito<val; ito++)
            {
                getVCenter(acc_out[cptchildV]) +=  cross(omega, cross(omega,(pointsR0[cptchildV]).getCenter()) );
                cptchildV++;
            }
        }
        break;

    default:
        if (repartition.getValue().size() != v_in.size())
        {
            serr<<"Error : mapping dofs repartition.getValue() is not correct"<<sendl;
            dAcc_out.endEdit();
            return;
        }
        cptchildV=0;
        for (unsigned int ifrom=0 ; ifrom<v_in.size() ; ifrom++)
        {
            omega = getVOrientation(v_in[ifrom]);

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                getVCenter(acc_out[cptchildV]) += cross(omega, cross(omega,(pointsR0[cptchildV]).getCenter()) );
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
	if (!getShow(this,vparams)) return;

    const typename Out::VecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const defaulttype::Vector3& sizes = defaulttype::Vector3(axisLength.getValue(), axisLength.getValue(), axisLength.getValue());
    for (unsigned int i=0; i<x.size(); i++)
    {
        vparams->drawTool()->drawFrame(x[i].getCenter(), x[i].getOrientation(), sizes);
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
