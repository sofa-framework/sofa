#ifndef PREQUIVALENTSTIFFNESSFORCEFIELD_INL
#define PREQUIVALENTSTIFFNESSFORCEFIELD_INL

#include <sofa/component/forcefield/PREquivalentStiffnessForceField.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/core/behavior/MechanicalState.h>

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
    m_complianceFile(initData(&m_complianceFile, "complianceFile", "Filename of the file where is stored the compliance matrix"))
{
    const std::string& filename = m_complianceFile.getValue();

    std::filebuf filebuffer;

    // Read compliance matrix from file
    if( filebuffer.open(filename.c_str(), std::ios_base::in) )
    {
        std::istream file(&filebuffer);
        file >> m_complianceMat;
        m_complianceMat.invert(m_CInv);
    }
    else
    {
        serr << "Cannot find compliance matrix file : " << filename << sendl;
    }

    // getting mstate bound to this forcefield
    sofa::core::behavior::MechanicalState<DataTypes>* mstate;
    this->getContext()->get(mstate, core::objectmodel::BaseContext::Local);

    if( mstate->getSize() > 2 )
        serr << "Error : context mstate has more than two frames" << sendl;

    // Read rest positions
    helper::WriteAccessor<VecCoord> restPosWriter(m_restPos);
    helper::ReadAccessor<DataVecCoord> restPosReader(*mstate->read(core::ConstVecCoordId::restPosition()));
    restPosWriter.resize(restPosWriter.size());

    std::copy(restPosReader.begin(), restPosReader.end(), restPosWriter.begin());
}

template<typename DataTypes>
PREquivalentStiffnessForceField<DataTypes>::~PREquivalentStiffnessForceField()
{

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::bwdInit()
{

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addForce(const MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v)
{
    const VecCoord& X = x.getValue();
    m_pos = x.getValue();
    VecDeriv& force = *f.beginEdit();

    const Pos& x0Current = X[0].getCenter();
    const Pos& x1Current = X[1].getCenter();

    const Pos& x0Rest = m_restPos[0].getCenter();
    const Pos& x1Rest = m_restPos[1].getCenter();

    const Quaternion& q0Current = X[0].getOrientation();
    const Quaternion& q1Current = X[1].getOrientation();
    const Quaternion& q0Rest = m_restPos[0].getOrientation();
    const Quaternion& q1Rest = m_restPos[1].getOrientation();

    // compute x1 local rigid position and rotation (can be precomputed)
    const Pos& x1th = q0Rest.inverseRotate(x1Rest - x0Rest);
    const Quaternion& q1th = Quaternion::identity();

    // compute x1 position w.r.t. x0 frame
    const Pos& x1l0Current = q0Current.inverseRotate(x1Current - x0Current);

    // compute the difference between theoritical and real positions and orientations
    const Pos& dx = x1th - x1l0Current;
//    Quaternion qDiff = q0Current.inverse() * q1th.inverse() * q1Current;
    Quaternion dummy;

    Vec3 dq = dummy.angularDisplacement(q0Current.inverse() * q1Current, q1th);


//    Real angleDiff;

//    qDiff.quatToAxis(dq, angleDiff);

//    dq.normalize();
//    dq *= angleDiff;

    Vec6 dX1(dx, dq);

    // compute x1 forces in x0's frame and rotate them back to global coordinates
    Vec6 tmp = m_CInv*dX1;
    Vec3 F(tmp(0), tmp(1), tmp(2)), r(tmp(3), tmp(4), tmp(5));
    F = q0Current.rotate(F);
    r = q0Current.rotate(r);

    Vec6 f1(F, r);

    // compute transport matrix
    Vec3 p0p1 = q0Current.inverseRotate(x1Current - x0Current);
    m_H = -Mat66::Identity();
    m_H(3, 1) = p0p1.z();
    m_H(3, 2) = -p0p1.y();
    m_H(4, 0) = -p0p1.z();
    m_H(4, 2) = p0p1.x();
    m_H(5, 0) = p0p1.y();
    m_H(5,1) = -p0p1.x();

    // compute x0 forces in x0's frame using transport
    tmp = m_H*tmp;

    F = Vec3(tmp(0), tmp(1), tmp(2));
    r = Vec3(tmp(3), tmp(4), tmp(5));

    F = q0Current.rotate(F);
    r = q0Current.rotate(r);

    Vec6 f0(F, r);

    for(size_t i = 0 ; i < 6 ; ++i)
    {
        force[0][i] += f0[i];
        force[1][i] += f1[i];
    }

    f.endEdit();

}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&  d_f , const DataVecDeriv&  d_x)
{
    const VecDeriv& dx = d_x.getValue();
    VecDeriv& dfdq = *d_f.beginEdit();

//    Mat12x12 K(0.0);
    m_K = Mat12x12(0.0);

    Mat66 Ht = m_H.transposed();
    Mat66 block = m_H * m_CInv * Ht;

    unsigned int rOffset, cOffset;
    rOffset = cOffset = 0u;

    for(int i = 0 ; i < 6 ; ++i)
        for(int j = 0 ; j < 6 ; ++j)
            m_K(i, j) = block(i, j);

    block = m_H*m_CInv;
    for(int i = 0 ; i < 6 ; ++i)
    {
        for(int j = 0 ; j < 6 ; ++j)
        {
            m_K(i, j+6) = block(i, j);
            m_K(i+6, j) = block(j, i);
        }
    }

    for(int i = 0 ; i < 6 ; ++i)
        for(int j = 0 ; j < 6 ; ++j)
            m_K(i+6, j+6) = m_CInv(i, j);

    Vec12 dq;
    Quaternion q0 = m_pos[0].getOrientation();

    for(size_t n = 0 ; n < dx.size() ; ++n)
    {
        Vec3 dp = q0.inverseRotate(dx[n].getVCenter());
        Vec3 dr = q0.inverseRotate(dx[n].getVOrientation());

        dq(6*n) = dp(0);
        dq(6*n+1) = dp(1);
        dq(6*n+2) = dp(2);
        dq(6*n+3) = dr(0);
        dq(6*n+4) = dr(1);
        dq(6*n+5) = dr(2);
    }

    Vec12 df = m_K*dq;

    for(size_t n = 0 ; n < dx.size() ; ++n)
    {
        Vec3 dfx(df(6*n), df(6*n+1), df(6*n+2));
        Vec3 dfr(df(6*n+3), df(6*n+4), df(6*n+5));

        dfx = q0.rotate(dfx);
        dfr = q0.rotate(dfr);

        df(6*n) = dfx(0);
        df(6*n+1) = dfx(1);
        df(6*n+2) = dfx(2);
        df(6*n+3) = dfx(0);
        df(6*n+4) = dfx(1);
        df(6*n+5) = dfx(2);
    }
}

template<typename DataTypes>
void PREquivalentStiffnessForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset)
{
    Mat33 block;
    Mat33 R0;
    m_pos[0].getOrientation().toMatrix(R0);

    // K is already built in the addDForce method.
    // We just need to rotate it by block

    // for each 3x3 block
    for(int i = 0 ; i < 4 ; ++i)
    {
        for(int j = 0 ; j < 4 ; ++j)
        {
            // get the sub matrix
            m_K.getsub(i*3, j*3, block);

            // apply the rotatation
            block = R0*block;

            // then store it in the final matrix
            for(int m = 0 ; m < 3 ; ++m)
            {
                for(int n = 0 ; n < 3 ; ++n)
                {
                    matrix->add(i*3+m, j*3+n, block(m, n));
                }
            }
        }
    }
}

} // forcefield

} // component

} // sofa

#endif // PREQUIVALENTSTIFFNESSFORCEFIELD_INL
