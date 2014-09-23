#ifndef PREQUIVALENTSTIFFNESSFORCEFIELD_INL
#define PREQUIVALENTSTIFFNESSFORCEFIELD_INL

#include <sofa/component/forcefield/PREquivalentStiffnessForceField.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/helper/RandomGenerator.h>

#include <fstream>
#include <iostream>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<typename DataTypes>
PREquivalentStiffnessForceField<DataTypes>::PREquivalentStiffnessForceField() : Inherit(),
    m_complianceFile(initData(&m_complianceFile, "complianceFile", "Filename of the file where is stored the compliance matrices"))
{

}

template<typename DataTypes>
PREquivalentStiffnessForceField<DataTypes>::~PREquivalentStiffnessForceField()
{

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::init()
{
    Inherit::init();
    const std::string& filename = m_complianceFile.getValue();

    std::filebuf filebuffer;

    std::cout << "Compliance file : " << filename << std::endl;

    // Read compliance matrix from file

    // getting mstate bound to this forcefield
    sofa::core::behavior::MechanicalState<DataTypes>* mstate;
    this->getContext()->get(mstate, core::objectmodel::BaseContext::Local);

//    if( mstate->getSize() > 2 )
//        serr << "Error : context mstate has more than two frames" << sendl;

    // Read rest positions
    helper::WriteAccessor<VecCoord> restPosWriter(m_restPos);
    helper::ReadAccessor<DataVecCoord> restPosReader(*mstate->read(core::ConstVecCoordId::restPosition()));
    size_t nFrames = restPosReader.size();
    restPosWriter.resize(nFrames);

    std::cout << "Copying positions" << std::endl;
    std::copy(restPosReader.begin(), restPosReader.end(), restPosWriter.begin());

    m_complianceMat.resize(nFrames-1);
    m_CInv.resize(nFrames-1);
    m_H.resize(nFrames-1);
    m_K.resize(nFrames-1);

    if( filebuffer.open(filename.c_str(), std::ios_base::in) )
    {
        std::istream file(&filebuffer);


        for(size_t n = 0 ; n < nFrames-1 ; ++n)
        {
            if( ! (file >> m_complianceMat[n]) )
            {
                serr << "Unable to read compliance matrix for frames [" << n << " | " << n+1 << "]" << sendl;
//                std::cout << m_complianceMat[n] << std::endl;
            }
            else
            {
                std::cout << "Inverting compliance #" << n << std::endl;
//                m_complianceMat[n].invert(m_CInv[n]);
                m_CInv[n].invert(m_complianceMat[n]);
                std::cout << "Cinv : " << m_CInv[n] << std::endl;
            }
        }
        filebuffer.close();
    }
    else
    {
        serr << "Cannot find compliance matrices file : " << filename << sendl;
    }
}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addForce(const MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v)
{
    const VecCoord& X = x.getValue();
    m_pos = x.getValue();
    VecDeriv& force = *f.beginEdit();
    size_t nFrames = X.size();

//    std::cout << "Number of frames : " << nFrames << std::endl;

    for(size_t n = 0 ; n < nFrames-1 ; ++n)
    {

        const Pos& x0Current = X[n+0].getCenter();
        const Pos& x1Current = X[n+1].getCenter();

        const Pos& x0Rest = m_restPos[n+0].getCenter();
        const Pos& x1Rest = m_restPos[n+1].getCenter();

        const Quaternion& q0Current = X[n+0].getOrientation();
        const Quaternion& q1Current = X[n+1].getOrientation();
        const Quaternion& q0Rest = m_restPos[n+0].getOrientation();
        const Quaternion& q1Rest = m_restPos[n+1].getOrientation();

        // compute x1 local rigid position and rotation (can be precomputed)
        const Pos& x1th = q0Rest.inverseRotate(x1Rest - x0Rest);
        const Quaternion& q1th = Quaternion::identity(); // currently q1th is the same

        // compute x1 position w.r.t. x0 frame
        const Pos& x1l0Current = q0Current.inverseRotate(x1Current - x0Current);

        // compute the difference between theoritical and real positions and orientations
        const Pos& dx = x1th - x1l0Current;
    //    Quaternion qDiff = q0Current.inverse() * q1th.inverse() * q1Current;
        Quaternion dummy;

        // compute rotation difference between rigid and real motion
        Vec3 dq = -dummy.angularDisplacement(q0Current.inverse() * q1Current, q1th);


    //    Real angleDiff;

    //    qDiff.quatToAxis(dq, angleDiff);

    //    dq.normalize();
    //    dq *= angleDiff;

        Vec6 dX1(dx, dq);

        // compute x1 forces in x0's frame and rotate them back to global coordinates
        Vec6 tmp = m_CInv[n]*dX1;
        Vec3 F(tmp(0), tmp(1), tmp(2)), r(tmp(3), tmp(4), tmp(5));
        F = q0Current.rotate(F); // shouldn't that be inverseRotate ?
        r = q0Current.rotate(r);

        Vec6 f1(F, r);

        // compute transport matrix
        Vec3 p0p1 = q0Current.inverseRotate(x1Current - x0Current);
        Mat66 H = Mat66::Identity();
        H(3, 1) = -p0p1.z();
        H(3, 2) = p0p1.y();
        H(4, 0) = p0p1.z();
        H(4, 2) = -p0p1.x();
        H(5, 0) = -p0p1.y();
        H(5,1) = p0p1.x();

        tmp = -H*tmp;

//        m_H[n] = H;0101

        Mat66 block = H*m_CInv[n];
        m_K[n].clear();
        m_K[n].setsub(0, 6, block);
        m_K[n].setsub(6, 0, block.transposed());

        block = block*H.transposed();
        m_K[n].setsub(0, 0, block);
        m_K[n].setsub(6, 6, m_CInv[n]);

        Mat33 Rn;
        X[n].getOrientation().toMatrix(Rn);
        Mat12x12 R(.0);
        R.setsub(0, 0, Rn);
        R.setsub(3, 3, Rn);
        R.setsub(6, 6, Rn);
        R.setsub(9, 9, Rn);

        m_K[n] = R * m_K[n] * R.transposed(); // modified : wasn't negated

        // compute x0 forces in x0's frame using transport

        F = Vec3(tmp(0), tmp(1), tmp(2));
        r = Vec3(tmp(3), tmp(4), tmp(5));

        F = q0Current.rotate(F);
        r = q0Current.rotate(r);

        Vec6 f0(F, r);

        force[n+0] += f0;
        force[n+1] += f1;

//        for(size_t i = 0 ; i < 6 ; ++i)
//        {
//            force[0][i] += f0[i];
//            force[1][i] += f1[i];
//        }
    }

//    std::cout << "f" << std::endl;
//    for(size_t n = 0 ; n < nFrames; ++n)
//    {
//        std::cout << force[n] << std::endl;
//    }
//    std::cout << std::endl;
//    std::cout << std::endl;

    f.endEdit();

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&  d_f , const DataVecDeriv&  d_x)
{
    const VecDeriv& dx = d_x.getValue();
    VecDeriv& dfdq = *d_f.beginEdit();
    const size_t nFrames = dx.size();

    VecCoord displaced(dx.size());

    const Real epsilon = 1e-2;

    for(size_t n = 0 ; n < nFrames-1 ; ++n)
    {
        Vec12 dq;
        // convert to Vec6 using getVAll() then grab the pointer
        Real* dq0 = dx[n+0].getVAll().ptr();
        Real* dq1 = dx[n+1].getVAll().ptr();

        // concatenate the two Vec6
        std::copy(dq0, dq0+6, dq.ptr());
        std::copy(dq1, dq1+6, dq.ptr()+6);

        Vec12 df = m_K[n]*dq;

        // separate force vector
        Vec6 df0(df.ptr());
        Vec6 df1(df.ptr()+6);

        dfdq[n+0] += df0;
        dfdq[n+1] += df1;

    }

    displaceFrames(m_pos, displaced, dx, epsilon);
    std::cout << displaced << std::endl;

    VecDeriv fq(dx.size()), fqdq(dx.size());
    helper::vector<Vec6> testDf(dx.size());

    computeForce(m_pos, m_restPos, fq);
    computeForce(displaced, m_restPos, fqdq);

//    fq = dataFq.getValue();
//    fqdq = dataFqdq.getValue();

    for(size_t n = 0 ; n < dx.size() ; ++n)
    {
        testDf[n] = (fqdq[n].getVAll() - fq[n].getVAll())/epsilon;
    }


    std::cout << "Df" << std::endl;
    for(size_t n = 0 ; n < nFrames; ++n)
    {
        std::cout << n << " : " << dfdq[n] << std::endl;
        std::cout << n << " : " << testDf[n] << std::endl;
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;

//    m_K.clear();

//    for(size_t n = 0 ; n < nFrames-1 ; ++n)
//    {
//        Mat12x12 K(0.0);

//        Mat66 Ht = m_H[n].transposed();
//        Mat66 block = m_H[n] * m_CInv[n] * Ht;

//        unsigned int rOffset, cOffset;
//        rOffset = cOffset = 0u;

//        for(int i = 0 ; i < 6 ; ++i)
//            for(int j = 0 ; j < 6 ; ++j)
//                K(i, j) = block(i, j);

//        block = m_H[n]*m_CInv[0];
//        for(int i = 0 ; i < 6 ; ++i)
//        {
//            for(int j = 0 ; j < 6 ; ++j)
//            {
//                K(i, j+6) = block(i, j);
//                K(i+6, j) = block(j, i);
//            }
//        }

//        for(int i = 0 ; i < 6 ; ++i)
//            for(int j = 0 ; j < 6 ; ++j)
//                K(i+6, j+6) = m_CInv[n](i, j);

//        Vec12 dq;
//        Quaternion q0 = m_pos[n+0].getOrientation();

//        for(size_t i = 0 ; i < 2 ; ++i)
//        {
//            Vec3 dp = q0.inverseRotate(dx[n+i].getVCenter());
//            Vec3 dr = q0.inverseRotate(dx[n+i].getVOrientation());

//            dq(6*i) = dp(0);
//            dq(6*i+1) = dp(1);
//            dq(6*i+2) = dp(2);
//            dq(6*i+3) = dr(0);
//            dq(6*i+4) = dr(1);
//            dq(6*i+5) = dr(2);
//        }

//        Vec12 df = K*dq;

//        for(int i = 0 ; i < 2 ; ++i)
//        {

//            Vec3 dfx(df(6*i), df(6*i+1), df(6*i+2));
//            Vec3 dfr(df(6*i+3), df(6*i+4), df(6*i+5));
//            Deriv& forceDeriv = dfdq[n+i];

//            dfx = q0.rotate(dfx);
//            dfr = q0.rotate(dfr);

//            forceDeriv[6*i] += dfx(0);
//            forceDeriv[6*i+1] += dfx(1);
//            forceDeriv[6*i+2] += dfx(2);
//            forceDeriv[6*i+3] += dfr(0);
//            forceDeriv[6*i+4] += dfr(1);
//            forceDeriv[6*i+5] += dfr(2);
//        }

//        K.getsub(0, 0, block);
//        *m_K.wbloc(n+0, n+0, true) += block;

//        K.getsub(0, 6, block);
//        *m_K.wbloc(n+0, n+1, true) += block;

//        K.getsub(6, 0, block);
//        *m_K.wbloc(n+1, n+0, true) += block;

//        K.getsub(6, 6, block);
//        *m_K.wbloc(n+1, n+1, true) += block;

//    }
}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset)
{
    // Edit added forgotten kfactor
    if( CSRMatB66 * csrMat = dynamic_cast<CSRMatB66*>(matrix) )
    {
        for(size_t n = 0 ; n < m_K.size() ; ++n)
        {
            const Mat12x12& K = m_K[n];
            Mat66 block;

            K.getsub(0, 0, block);
            *csrMat->wbloc(n, n, true) += block*kFact;

            K.getsub(0, 6, block);
            *csrMat->wbloc(n, n+1, true) += block*kFact;

            K.getsub(6, 0, block);
            *csrMat->wbloc(n+1, n, true) += block*kFact;

            K.getsub(6, 6, block);
            *csrMat->wbloc(n+1, n+1, true) += block*kFact;
        }
    }
    else
    {
        unsigned int kOffset = offset;
        for(size_t n = 0 ; n < m_K.size() ; ++n)
        {
            const Mat12x12& K = m_K[n];

            for(int i = 0 ; i < 12 ; ++i)
            {
                for(int j = 0 ; j < 12 ; ++j)
                {
                    matrix->add(kOffset + i , kOffset + j, K(i,j)*kFact);
                }
            }
            kOffset += 6;
        }
    }

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::displaceFrames(const VecCoord &frames, PREquivalentStiffnessForceField::VecCoord &displaced, const VecDeriv &dq, const Real epsilon)
{
    sofa::helper::RandomGenerator rgen;
    for(size_t n = 0 ; n < displaced.size() ; ++n)
    {
        Deriv dqn = dq[n]*epsilon;
//        for(int i = 0 ; i < Deriv::total_size ; ++i)
//            dqn[i] = rgen.random(.0, 1.0)*epsilon;

        Quaternion q(dqn.getVOrientation(), dqn.getVOrientation().norm());
        q.normalize();
        displaced[n].getCenter() = frames[n].getCenter() + dqn.getVCenter();
        displaced[n].getOrientation() = frames[n].getOrientation() * q;
    }
}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::computeForce(const VecCoord& pos, const VecCoord& restPos, VecDeriv& f)
{
    size_t nFrames = pos.size();

//    std::cout << "Number of frames : " << nFrames << std::endl;

    for(size_t n = 0 ; n < nFrames-1 ; ++n)
    {

        const Pos& x0Current = pos[n+0].getCenter();
        const Pos& x1Current = pos[n+1].getCenter();

        const Pos& x0Rest = restPos[n+0].getCenter();
        const Pos& x1Rest = restPos[n+1].getCenter();

        const Quaternion& q0Current = pos[n+0].getOrientation();
        const Quaternion& q1Current = pos[n+1].getOrientation();
        const Quaternion& q0Rest = restPos[n+0].getOrientation();
        const Quaternion& q1Rest = restPos[n+1].getOrientation();

        // compute x1 local rigid position and rotation (can be precomputed)
        const Pos& x1th = q0Rest.inverseRotate(x1Rest - x0Rest);
        const Quaternion& q1th = Quaternion::identity(); // currently q1th is the same

        // compute x1 position w.r.t. x0 frame
        const Pos& x1l0Current = q0Current.inverseRotate(x1Current - x0Current);

        // compute the difference between theoritical and real positions and orientations
        const Pos& dx = x1th - x1l0Current;

        Quaternion dummy;

        // compute rotation difference between rigid and real motion
        Vec3 dq = -dummy.angularDisplacement(q0Current.inverse() * q1Current, q1th);

        Vec6 dX1(dx, dq);

        // compute x1 forces in x0's frame and rotate them back to global coordinates
        Vec6 tmp = m_CInv[n]*dX1;
        Vec3 F(tmp(0), tmp(1), tmp(2)), r(tmp(3), tmp(4), tmp(5));
        F = q0Current.rotate(F); // shouldn't that be inverseRotate ?
        r = q0Current.rotate(r);

        Vec6 f1(F, r);

        // compute transport matrix
        Vec3 p0p1 = q0Current.inverseRotate(x1Current - x0Current);
        Mat66 H = Mat66::Identity();
        H(3, 1) = -p0p1.z();
        H(3, 2) = p0p1.y();
        H(4, 0) = p0p1.z();
        H(4, 2) = -p0p1.x();
        H(5, 0) = -p0p1.y();
        H(5,1) = p0p1.x();

        tmp = -H*tmp;

        // compute x0 forces in x0's frame using transport

        F = Vec3(tmp(0), tmp(1), tmp(2));
        r = Vec3(tmp(3), tmp(4), tmp(5));

        F = q0Current.rotate(F);
        r = q0Current.rotate(r);

        Vec6 f0(F, r);

        f[n+0] += f0;
        f[n+1] += f1;

    }
}

} // forcefield

} // component

} // sofa

#endif // PREQUIVALENTSTIFFNESSFORCEFIELD_INL
