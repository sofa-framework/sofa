#ifndef FRAMEMAPPING_INL
#define FRAMEMAPPING_INL

#include "../deformationMapping/FrameMapping.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/Mapping.inl>

namespace sofa
{
namespace component
{

namespace mapping
{

using defaulttype::Rigid3fTypes;
using defaulttype::Vec3fTypes;

//template<> void FrameMapping<Rigid3fTypes, Vec3fTypes>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn);

template<class TIn, class TOut>
FrameMapping<TIn, TOut>::FrameMapping() : Inherit(),
    m_restFrame(initData(&m_restFrame, "restFrames", "restFrames")),
    m_restPos(initData(&m_restPos, "restPos", "restPos")),
    m_shapeFun(NULL),
    m_pointShapeFun(NULL),
    m_indices(initData(&m_indices, "indices", "Indices of neighbors")),
    m_w(initData(&m_w, "weights", "Weights of frames")),
    m_dw(initData(&m_dw, "gradWeights", "Gradient of weights")),
    m_ddw(initData(&m_ddw, "hessianWeights", "Hessian of weights"))
{

}

template<class TIn, class TOut>
FrameMapping<TIn, TOut>::~FrameMapping()
{

}

//template<class TIn, class TOut>
//void FrameMapping<TIn, TOut>::init()
//{
//    Inherit::init();
//}

template<class TIn, class TOut>
void FrameMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams /*params*/, Data<OutVecCoord>& out, const Data<InVecCoord>& in)
{
    std::cout << in.size() << std::endl;
    std::cout << out.size() << std::endl;
}

template<class TIn, class TOut>
void FrameMapping<TIn, TOut>::init(void)
{
    std::cout << "FRAMEMAPPING INIT" << std::endl;


    helper::ReadAccessor< Data<InVecCoord> > inFrame(*this->fromModel->read(core::ConstVecCoordId::restPosition()));
    helper::ReadAccessor< Data<OutVecCoord> > outPos(*this->toModel->read(core::ConstVecCoordId::restPosition()));
    std::cout << "In vel" << std::endl;
    helper::ReadAccessor< Data<InVecDeriv> > inFrameVel(*this->fromModel->read(core::ConstVecDerivId::velocity()));
    std::cout << "Out vel" << std::endl;
    helper::ReadAccessor<  Data<OutVecDeriv> > particleVel(*this->toModel->read(core::ConstVecDerivId::velocity()));
    helper::WriteAccessor< Data<InVecCoord> > restFrameAccess(m_restFrame);
    helper::WriteAccessor< Data<OutVecCoord> > restPosAccess(m_restPos);
    std::cout << "In vel access" << std::endl;
    helper::WriteAccessor< InVecDeriv > frameInitVelAccess(m_frameInitVel);
    std::cout << "Out vel access" << std::endl;
    helper::WriteAccessor< OutVecDeriv > particleInitVelAccess(m_particleInitVel);

    restFrameAccess.resize(inFrame.size());
    restPosAccess.resize(outPos.size());
    frameInitVelAccess.resize(inFrameVel.size());
    particleInitVelAccess.resize(particleVel.size());

    std::copy(inFrame.begin(), inFrame.end(), restFrameAccess.begin());
    std::copy(outPos.begin(), outPos.end(), restPosAccess.begin());
    std::copy(inFrameVel.begin(), inFrameVel.end(), frameInitVelAccess.begin());
    std::copy(particleVel.begin(), particleVel.end(), particleInitVelAccess.begin());

    Data<VMaterialToSpatial> M;

    helper::vector<defaulttype::Vec3f> framePos;
    for(int i = 0 ; i < m_restFrame.getValue().size() ; ++i)
        framePos.push_back(m_restFrame.getValue()[i].getCenter());

    if( ! m_shapeFun )
    {
        sout << "no shape fun" << sendl;
        this->getContext()->get(m_shapeFun, core::objectmodel::BaseContext::SearchParents);
    }

    if( ! m_shapeFun )
    {
        serr << "No shape function found " << sendl;
    }
    else
    {
        m_shapeFun->computeShapeFunction(*m_restPos.beginEdit(), *M.beginEdit(), *m_indices.beginEdit(), *m_w.beginEdit(), *m_dw.beginEdit(), *m_ddw.beginEdit());
    }

    if( ! m_pointShapeFun )
    {
        this->getContext()->get(m_pointShapeFun, core::objectmodel::BaseContext::SearchDown);
    }


    if( ! m_pointShapeFun )
    {
        serr << "No point shape fun" << sendl;
    }
    else
    {
        std::cout << m_pointShapeFun->getName() << std::endl;
        m_pointShapeFun->computeShapeFunction(framePos, *M.beginEdit(), *m_pointsToFramesIndices.beginEdit(), *m_pointsW.beginEdit(), *m_pointsDw.beginEdit(), *m_pointsDdw.beginEdit());
    }

    Inherit::init();
}

template<>
void FrameMapping< Rigid3fTypes, Vec3fTypes >::apply(const core::MechanicalParams* mparams /* params */, Data<OutVecCoord>& out, const Data<InVecCoord>& in)
{
//    std::cout << in.getValue().size() << std::endl;
//    std::cout << out.getValue().size() << std::endl;

    InVecCoord frames = in.getValue();
    InVecCoord restFrames = m_restFrame.getValue();
    OutVecCoord particleRestPos = m_restPos.getValue();
    vector<VRef> neighborList = m_indices.getValue();
    vector<VReal> weights = m_w.getValue();
    helper::WriteAccessor< Data<OutVecCoord> > output(out);
    std::cout << "FrameMapping apply()" << std::endl;
    std::cout << "restPos size : " << m_restPos.getValue().size() << std::endl;
    std::cout << "indices size : " << m_indices.getValue().size() << std::endl;
//    if(m_indices.getValue().size())
//    {
//        for(int i = 0 ; i < m_indices.getValue().size() ; ++i )
//            std::cout << "indices[" << i << "] size : " << (m_indices.getValue())[i].size() << std::endl;
//    }

    for(int i = 0 ; i < neighborList.size() ; ++i)
    {
        OutCoord p0 = particleRestPos[i];
        VRef particleNeighbors = neighborList[i];
        OutCoord outputCoord;

        for(int n = 0 ; n < particleNeighbors.size() ; ++ n)
        {
            InCoord frameRestPos = restFrames[particleNeighbors[n]];
            OutCoord localRestPos =  frameRestPos.getOrientation().inverseRotate(p0 - frameRestPos.getCenter());


            // OutCoord frameDisplacement = frameRestPos.getCenter() - frames[particleNeighbors[n]].getCenter();
            //OutCoord localPos = p0 - framePos.getCenter();
            InCoord::Quat rotation = frames[particleNeighbors[n]].getOrientation();


            outputCoord += weights[i][n]*(frames[particleNeighbors[n]].getCenter()+ rotation.rotate(localRestPos) );
        }

        output[i] = outputCoord;
    }

//    for(int i = 0 ; i < m_restPos.getValue().size() ; ++i)
//    {
//        VRef neighbors = m_indices.getValue()[i];

//        for(int n = 0 ; n < neighbors.size() ; ++i)
//            std::cout << neighbors[i] << " " ;

//        std::cout << std::endl;
//    }


}

template<>
void FrameMapping<Rigid3fTypes, Vec3fTypes>::applyJ(const core::MechanicalParams* mparams /* PARAMS FIRST */, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in)
{

    helper::ReadAccessor< Data<InVecCoord> > framePos(*this->fromModel->read(core::ConstVecCoordId::restPosition()));
    helper::ReadAccessor< Data<OutVecCoord> > particlePos(*this->toModel->read(core::ConstVecCoordId::restPosition()));
    InVecDeriv frameVel = in.getValue();
    OutVecDeriv particleVel = out.getValue();
    vector<VRef> neighborList = m_indices.getValue();
    vector<VReal> weights = m_w.getValue();

    for(int i = 0 ; i < neighborList.size() ; ++i)
    {
        OutDeriv pv0 = m_particleInitVel[i];
        OutDeriv pv = particleVel[i];
        OutCoord pp = particlePos[i];
        VRef neighbors = neighborList[i];
        OutDeriv outVel;

        for(int n = 0 ; n < neighbors.size() ; ++n)
        {
            InDeriv fv0 = m_frameInitVel[neighbors[n]];
            InDeriv fv = frameVel[neighbors[n]];

            InCoord fp = framePos[neighbors[n]];
            OutCoord localPPos = fp.getOrientation().inverseRotate(pp - fp.getCenter());

            outVel += weights[i][n]*fv.getVOrientation().cross(localPPos);
        }

        particleVel[i] += outVel;
    }

}

template<>
void FrameMapping<Rigid3fTypes, Vec3fTypes>::applyJT(const core::MechanicalParams* mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in)
{

}

} //namespace mapping
} //namespace component
} //namespace sofa

#endif // FRAMEMAPPING_INL
