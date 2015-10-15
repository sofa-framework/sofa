/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_PCAONRIGIDFRAME_INL
#define SOFA_COMPONENT_MAPPING_PCAONRIGIDFRAME_INL

#include <SofaMiscMapping/PCAOnRigidFrameMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Quat.h>

#include <sofa/simulation/common/Simulation.h>

#include <string.h>
#include <iostream>


/*!
*   This mapping is derived from the RigidMapping. The difference is :
*   In the RigidMapping, the rigid is considered like a perfect rigid
*   In this one, there is a local deformation (supposed small) based on the linear combination of basis vectors.
*
*	Mapping : p = R (pref + Uw ) + t = R.deformedpoints + t = rotatedpoints + t
*	- R,t = rigid dofs (input root dofs)
*	- w = 1D vector of weights (input dofs)
*	- U = basis displacement vectors
*	- pref = out rest positions = mean shape
*	- p = out positions
*	Jacobian : dp = [ -rotatedpoints_x  I  RU ] [omega v dw ]^T
*/

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TInRoot, class TOut>
PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::PCAOnRigidFrameMapping ()
    : basis(initData(&basis,"basis","Basis of deformation modes."))
    , index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) )
    , indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") )
    , repartition ( initData ( &repartition,"repartition","number of dest dofs per entry rigid dof" ) )
{
    maskFrom = NULL;
    maskTo = NULL;
}



template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::init()
{
    Inherit::init();
    if(!this->fromModels1.empty())
    {
        if (core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *>(this->fromModels1.get(0)))
            maskFrom = &stateFrom->forceMask;
    }
    if(!this->toModels.empty())
    {
        if (core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *>(this->toModels.get(0)))
            maskTo = &stateTo->forceMask;
    }

    if(this->getFromModels1().empty())
    {
        serr << "Error while iniatilizing ; input Model not found" << sendl;
        return;
    }

    if(this->getToModels().empty())
    {
        serr << "Error while iniatilizing ; output Model not found" << sendl;
        return;
    }

    m_fromModel = this->getFromModels1()[0];
    m_toModel = this->getToModels()[0];

    //Root
    if(!this->getFromModels2().empty())
    {
        m_fromRootModel = this->getFromModels2()[0];
        sout << "Root Model found : Name = " << m_fromRootModel->getName() << sendl;
    }
}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::clear(int /*reserve*/)
{

}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::setRepartition(unsigned int value)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::setRepartition(sofa::helper::vector<unsigned int> /*values*/)
{

}

template<class DataTypes>
const typename DataTypes::VecCoord& M_getX0(core::behavior::MechanicalState<DataTypes>* model)
{
    return model->read(core::ConstVecCoordId::restPosition())->getValue();
}

template<class DataTypes>
const typename DataTypes::VecCoord* M_getX0(core::State<DataTypes>* /*model*/)
{
    return NULL;
}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::apply( typename Out::VecCoord& out, const typename In::VecCoord& inWeights, const typename InRoot::VecCoord* inRigid  )
{
    unsigned int nbmodes = inWeights.size();
    unsigned int nbpoints = basis.getValue().size()/(1+nbmodes);
    out.resize(nbpoints);

    // deformedpoints = pref + Uw
    deformedPoints.resize(nbpoints);
    sofa::helper::ReadAccessor<Data< OutVecCoord > > _basis = basis;
    for(unsigned int i=0; i<nbpoints; i++)
    {
        deformedPoints[i] = _basis[i];
        for(unsigned int j=0; j<nbmodes; j++)
            deformedPoints[i]+=_basis[(j+1)*nbpoints+i]*inWeights[j][0];
    }

    // rotatedpoints = R.deformedpoints
    // out = rotatedpoints + t
    rotatedPoints.resize(nbpoints);

    if (m_fromRootModel)
    {
        unsigned int nbrigids = (*inRigid).size();
        rootX.resize(nbrigids); for(unsigned int i=0; i<nbrigids; i++) rootX[i]=(*inRigid)[i];

        switch (repartition.getValue().size())
        {

        case 0 :
        {
            unsigned int val = (indexFromEnd.getValue()) ? rootX.size() - 1 - index.getValue() : index.getValue();

            Coord translation = rootX[val].getCenter();
            sofa::defaulttype::Quat rot = rootX[val].getOrientation();

            for(unsigned int i=0; i<nbpoints; i++)
            {
                rotatedPoints[i] = rot.rotate(deformedPoints[i]);
                out[i] = rotatedPoints[i] + translation;
            }
        }
        break;

        case 1 ://one value specified : uniform repartition mapping on the input dofs
        {
            unsigned int val = repartition.getValue()[0];
            unsigned int cptOut = 0;
            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                Coord translation = rootX[ifrom].getCenter();
                sofa::defaulttype::Quat rot = rootX[ifrom].getOrientation();

                for(unsigned int ito=0; ito<val; ito++)
                {
                    rotatedPoints[cptOut] = rot.rotate(deformedPoints[cptOut]);
                    out[cptOut] = rotatedPoints[cptOut] + translation;
                    cptOut++;
                }
            }
        }
        break;

        default :
        {
            if (repartition.getValue().size() != rootX.size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }
            unsigned int cptOut=0;

            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                Coord translation = rootX[ifrom].getCenter();
                sofa::defaulttype::Quat rot = rootX[ifrom].getOrientation();

                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    rotatedPoints[cptOut] = rot.rotate(deformedPoints[cptOut]);
                    out[cptOut] = rotatedPoints[cptOut] + translation;
                    cptOut++;
                }
            }
        }

        }
    }
    else  // no m_fromRootModel found => mapping is identity !
    {
        for(unsigned int i=0; i<nbpoints; i++) out[i] = rotatedPoints[i] = deformedPoints[i];
    }

}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJ( typename Out::VecDeriv&  out , const typename In::VecDeriv& inWeights , const typename InRoot::VecDeriv* inRigid)
{
    // Jacobian : dp = [ -rotatedpoints_x  I  RU ] [omega v w' ]^T

    unsigned int nbmodes = inWeights.size();
    unsigned int nbpoints = rotatedPoints.size();
    out.resize(nbpoints);

    // deformedpoints' = Uw'
    deformedPoints.resize(nbpoints);
    sofa::helper::ReadAccessor<Data< OutVecCoord > > _basis = basis;
    for(unsigned int i=0; i<nbpoints; i++)
    {
        deformedPoints[i] = OutCoord();
        for(unsigned int j=0; j<nbmodes; j++)
            deformedPoints[i]+=_basis[(j+1)*nbpoints+i]*inWeights[j][0];
    }

    if (m_fromRootModel)
    {
        Deriv v,omega;
        sofa::defaulttype::Quat rot;
        unsigned int cptOut;
        unsigned int val;

        switch (repartition.getValue().size())
        {
        case 0:
            if (indexFromEnd.getValue())	val=rootX.size() - 1 - index.getValue();
            else val=index.getValue();

            v = getVCenter((*inRigid)[val]);
            omega = getVOrientation((*inRigid)[val]);
            rot = rootX[val].getOrientation();

            for(unsigned int i=0; i<nbpoints; i++)
            {
                out[i] = cross(omega,rotatedPoints[i]);
                out[i] += rot.rotate(deformedPoints[i]);
                out[i] += v;
            }
            break;

        case 1 ://one value specified : uniform repartition mapping on the input dofs
            val = repartition.getValue()[0];
            cptOut=0;
            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                v = getVCenter((*inRigid)[ifrom]);
                omega =	getVOrientation((*inRigid)[ifrom]);
                rot = rootX[ifrom].getOrientation();

                for(unsigned int ito=0; ito<val; ito++)
                {
                    out[cptOut] = cross(omega,rotatedPoints[cptOut]);
                    out[cptOut] += rot.rotate(deformedPoints[cptOut]);
                    out[cptOut] += v;
                    cptOut++;
                }
            }
            break;

        default :
            if (repartition.getValue().size() != rootX.size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }
            cptOut=0;

            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                v = getVCenter((*inRigid)[ifrom]);
                omega =	getVOrientation((*inRigid)[ifrom]);
                rot = rootX[ifrom].getOrientation();

                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    out[cptOut] = cross(omega,rotatedPoints[cptOut]);
                    out[cptOut] += rot.rotate(deformedPoints[cptOut]);
                    out[cptOut] += v;
                    cptOut++;
                }
            }
            break;
        }
    }
    else // no root model!
    {
        for(unsigned int i=0; i<nbpoints; i++)	out[i] = deformedPoints[i];
    }
}


template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT( typename In::VecDeriv& outWeights, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outRigid)
{
    // Jacobian^T : [omega v dw ]^T = [ -rotatedpoints_x  I  RU ]^T dp

    sofa::helper::ReadAccessor<sofa::core::objectmodel::Data< OutVecCoord > > _basis = basis;
    unsigned int nbmodes = outWeights.size();
    unsigned int nbpoints = basis.getValue().size()/(1+nbmodes);

    //if (in.size() != rotatedPoints.size())
    //{
    //	const InDataVecCoord* xfromData = m_fromModel->read(core::ConstVecCoordId::position());
    //	const InVecCoord xfrom = xfromData->getValue();
    //	OutDataVecCoord* xtoData = m_toModel->write(core::VecCoordId::position());
    //	OutVecCoord &xto = *xtoData->beginEdit();
    //	apply(xto, xfrom, (m_fromRootModel==NULL ? NULL : m_fromRootModel->read(sofa::core::ConstVecCoordId::position())->getValue()));
    //	xtoData->endEdit();
    //}

    Coord p; // sum Ri^T dpi
    if (m_fromRootModel)
    {
        Deriv v,omega;
        unsigned int cptOut;
        unsigned int val;

        switch (repartition.getValue().size())
        {
        case 0:

            if (indexFromEnd.getValue())	val=rootX.size() - 1 - index.getValue();
            else val=index.getValue();

            for(unsigned int i=0; i<nbpoints; i++)
            {
                v += in[i];
                omega += cross(rotatedPoints[i],in[i]);
                p = rootX[val].getOrientation().inverseRotate(in[i]);
                for(unsigned int j=0; j<nbmodes; j++)
                    outWeights[j][0]+= dot(_basis[(j+1)*nbpoints+i],p);
            }
            getVCenter((*outRigid)[val]) += v;
            getVOrientation((*outRigid)[val]) += omega;
            break;

        case 1 ://one value specified : uniform repartition mapping on the input dofs
            val = repartition.getValue()[0];
            cptOut=0;
            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                v = omega = Deriv();
                for(unsigned int ito=0; ito<val; ito++)
                {
                    v += in[cptOut];
                    omega += cross(rotatedPoints[cptOut],in[cptOut]);
                    p = rootX[ifrom].getOrientation().inverseRotate(in[cptOut]);
                    for(unsigned int j=0; j<nbmodes; j++)
                        outWeights[j][0]+= dot(_basis[(j+1)*nbpoints+cptOut],p);
                    cptOut++;
                }
                getVCenter((*outRigid)[ifrom]) += v;
                getVOrientation((*outRigid)[ifrom]) += omega;
            }
            break;

        default :
            if (repartition.getValue().size() != rootX.size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }
            cptOut=0;

            for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
            {
                v = omega = Deriv();
                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    v += in[cptOut];
                    omega += cross(rotatedPoints[cptOut],in[cptOut]);
                    p = rootX[ifrom].getOrientation().inverseRotate(in[cptOut]);
                    for(unsigned int j=0; j<nbmodes; j++)
                        outWeights[j][0]+= dot(_basis[(j+1)*nbpoints+cptOut],p);
                    cptOut++;
                }
                getVCenter((*outRigid)[ifrom]) += v;
                getVOrientation((*outRigid)[ifrom]) += omega;
            }
            break;
        }
    }
    else
    {
        for(unsigned int i=0; i<nbpoints; i++)
            for(unsigned int j=0; j<nbmodes; j++)
                outWeights[j][0]+= dot(_basis[(j+1)*nbpoints+i],in[i]);
    }
}

template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT(typename In::MatrixDeriv&  outWeights , const typename Out::MatrixDeriv&  in , typename InRoot::MatrixDeriv*  outRigid)
{
    // Jacobian^T : [omega v dw ]^T = [ -rotatedpoints_x  I  RU ]^T dp

    sofa::helper::ReadAccessor<sofa::core::objectmodel::Data< OutVecCoord > > _basis = basis;
    unsigned int nbmodes = outWeights.size();

    Coord p; // sum Ri^T dpi

    if (m_fromRootModel)
    {
        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != in.end(); ++rowIt)
        {
            typename In::MatrixDeriv::RowIterator oW = outWeights.writeLine(rowIt.index());
            typename InRoot::MatrixDeriv::RowIterator oR = outRigid->writeLine(rowIt.index());

            if(repartition.getValue().size()==0)
            {
                Vector v, omega;
                unsigned int val;
                if (indexFromEnd.getValue())	val=rootX.size() - 1 - index.getValue();
                else val=index.getValue();

                for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
                {
                    v += colIt.val();
                    omega += cross(rotatedPoints[colIt.index()],colIt.val());
                    p = rootX[val].getOrientation().inverseRotate(colIt.val());
                    for(unsigned int j=0; j<nbmodes; j++)
                    {
                        const InDeriv d(dot(_basis[(j+1)*rowIt.end().index()+colIt.index()],p));
                        oW.addCol(j,d);
                    }
                }
                const InRootDeriv result(v, omega);
                oR.addCol(val, result);
            }
            else if(repartition.getValue().size()==1)
            {
                unsigned int val = repartition.getValue()[0];
                typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

                for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
                {
                    Vector v, omega;
                    for(unsigned int ito=0; ito<val; ito++)
                    {
                        v += colIt.val();
                        omega += cross(rotatedPoints[colIt.index()],colIt.val());
                        p = rootX[ifrom].getOrientation().inverseRotate(colIt.val());
                        for(unsigned int j=0; j<nbmodes; j++)
                        {
                            const InDeriv d(dot(_basis[(j+1)*rowIt.end().index()+colIt.index()],p));
                            oW.addCol(j,d);
                        }
                        colIt++;
                    }
                    const InRootDeriv result(v, omega);
                    oR.addCol(ifrom, result);
                }
            }
            else
            {
                if (repartition.getValue().size() != rootX.size())
                {
                    serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                    return;
                }

                typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

                for (unsigned int ifrom=0 ; ifrom<rootX.size() ; ifrom++)
                {
                    Vector v, omega;
                    for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                    {
                        v += colIt.val();
                        omega += cross(rotatedPoints[colIt.index()],colIt.val());
                        p = rootX[ifrom].getOrientation().inverseRotate(colIt.val());
                        for(unsigned int j=0; j<nbmodes; j++)
                        {
                            const InDeriv d(dot(_basis[(j+1)*rowIt.end().index()+colIt.index()],p));
                            oW.addCol(j,d);
                        }
                        colIt++;
                    }
                    const InRootDeriv result(v, omega);
                    oR.addCol(ifrom, result);
                }
            }
        }
    }
    else
    {
        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != in.end(); ++rowIt)
        {
            typename In::MatrixDeriv::RowIterator oW = outWeights.writeLine(rowIt.index());
            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
                for(unsigned int j=0; j<nbmodes; j++)
                {
                    const InDeriv d(dot(_basis[(j+1)*rowIt.end().index()+colIt.index()],colIt.val()));
                    oW.addCol(j, d);
                }
        }
    }

}


template <class TIn, class TInRoot, class TOut>
void PCAOnRigidFrameMapping<TIn, TInRoot, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;
    std::vector< sofa::defaulttype::Vector3 > points;
    sofa::defaulttype::Vector3 point;

    const typename Out::VecCoord& x = m_toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        point = Out::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 7, sofa::defaulttype::Vec<4,float>(1,1,0,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PCAONRIGIDFRAME_INL
