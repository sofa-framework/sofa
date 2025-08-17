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
#define SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_CPP
#include <sofa/component/mechanicalload/SurfacePressureForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::mechanicalload
{

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addDForce(const core::MechanicalParams* mparams,
                                                                    DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    auto df = sofa::helper::getWriteOnlyAccessor(d_df);
    const VecDeriv& dx = d_dx.getValue();


    for (unsigned int i = 0; i < derivTriNormalIndices.size(); i++)
    {
        for (unsigned int j = 0; j < derivTriNormalIndices[i].size(); j++)
        {
            const unsigned int v = derivTriNormalIndices[i][j];
            df[i].getVCenter() += (derivTriNormalValues[i][j] * dx[v].getVCenter()) * kFactor;
        }
    }

}

template <>
SurfacePressureForceField<defaulttype::Rigid3Types>::Real SurfacePressureForceField<defaulttype::Rigid3Types>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;

    Real volume = 0;

    unsigned int nTriangles = 0;
    const VecIndex& triangleIndices = d_triangleIndices.getValue();
    if (!triangleIndices.empty())
    {
        nTriangles = triangleIndices.size();
    }
    else
    {
        nTriangles = m_topology->getNbTriangles();
    }

    unsigned int triangleIdx = 0;
    for (unsigned int i = 0; i < nTriangles; i++)
    {
        if (!triangleIndices.empty())
        {
            triangleIdx = triangleIndices[i];
        }
        else
        {
            triangleIdx = i;
        }
        Triangle t = m_topology->getTriangle(triangleIdx);
        const defaulttype::Rigid3Types::CPos a = x[t[0]].getCenter();
        const defaulttype::Rigid3Types::CPos b = x[t[1]].getCenter();
        const defaulttype::Rigid3Types::CPos c = x[t[2]].getCenter();
        volume += dot(cross(a, b), c);
    }

    unsigned int nQuads = 0;
    const VecIndex& quadIndices = d_quadIndices.getValue();
    if (!quadIndices.empty())
    {
        nQuads = quadIndices.size();
    }
    else
    {
        nQuads = m_topology->getNbQuads();
    }

    unsigned int quadIdx = 0;
    for (unsigned int i = 0; i < nQuads; i++)
    {
        if (!quadIndices.empty())
        {
            quadIdx = quadIndices[i];
        }
        else
        {
            quadIdx = i;
        }
        Quad q = m_topology->getQuad(quadIdx);
        const defaulttype::Rigid3Types::CPos a = x[q[0]].getCenter();
        const defaulttype::Rigid3Types::CPos b = x[q[1]].getCenter();
        const defaulttype::Rigid3Types::CPos c = x[q[2]].getCenter();
        const defaulttype::Rigid3Types::CPos d = x[q[3]].getCenter();
        volume += dot(cross(a, b), c);
        volume += dot(cross(a, c), d);
    }

    // Divide by 6 when computing tetrahedron volume
    return volume / 6.0;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
    Triangle t = m_topology->getTriangle(triId);

    defaulttype::Rigid3Types::CPos ab = x[t[1]].getCenter() - x[t[0]].getCenter();
    defaulttype::Rigid3Types::CPos ac = x[t[2]].getCenter() - x[t[0]].getCenter();
    defaulttype::Rigid3Types::CPos bc = x[t[2]].getCenter() - x[t[1]].getCenter();

    defaulttype::Rigid3Types::CPos p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));


    if (computeDerivatives)
    {
        Mat33 DcrossDA;
        DcrossDA(0,0) = 0;
        DcrossDA(0,1) = -bc[2];
        DcrossDA(0,2) = bc[1];
        DcrossDA(1,0) = bc[2];
        DcrossDA(1,1) = 0;
        DcrossDA(1,2) = -bc[0];
        DcrossDA(2,0) = -bc[1];
        DcrossDA(2,1) = bc[0];
        DcrossDA(2,2) = 0;

        Mat33 DcrossDB;
        DcrossDB(0,0) = 0;
        DcrossDB(0,1) = ac[2];
        DcrossDB(0,2) = -ac[1];
        DcrossDB(1,0) = -ac[2];
        DcrossDB(1,1) = 0;
        DcrossDB(1,2) = ac[0];
        DcrossDB(2,0) = ac[1];
        DcrossDB(2,1) = -ac[0];
        DcrossDB(2,2) = 0;


        Mat33 DcrossDC;
        DcrossDC(0,0) = 0;
        DcrossDC(0,1) = -ab[2];
        DcrossDC(0,2) = ab[1];
        DcrossDC(1,0) = ab[2];
        DcrossDC(1,1) = 0;
        DcrossDC(1,2) = -ab[0];
        DcrossDC(2,0) = -ab[1];
        DcrossDC(2,1) = ab[0];
        DcrossDC(2,2) = 0;

        for (unsigned int j = 0; j < 3; j++)
        {
            derivTriNormalValues[t[j]].push_back(DcrossDA * (pressure / static_cast<Real>(6.0)));
            derivTriNormalValues[t[j]].push_back(DcrossDB * (pressure / static_cast<Real>(6.0)));
            derivTriNormalValues[t[j]].push_back(DcrossDC * (pressure / static_cast<Real>(6.0)));

            derivTriNormalIndices[t[j]].push_back(t[0]);
            derivTriNormalIndices[t[j]].push_back(t[1]);
            derivTriNormalIndices[t[j]].push_back(t[2]);
        }
    }


    if (d_mainDirection.getValue().getVCenter() != defaulttype::Rigid3Types::CPos())
    {
        defaulttype::Rigid3Types::CPos n = ab.cross(ac);
        n.normalize();
        const Real scal = n * d_mainDirection.getValue().getVCenter();
        p *= fabs(scal);
    }

    f[t[0]].getVCenter() += p;
    f[t[1]].getVCenter() += p;
    f[t[2]].getVCenter() += p;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
    Quad q = m_topology->getQuad(quadId);

    const defaulttype::Rigid3Types::CPos ab = x[q[1]].getCenter() - x[q[0]].getCenter();
    const defaulttype::Rigid3Types::CPos ac = x[q[2]].getCenter() - x[q[0]].getCenter();
    const defaulttype::Rigid3Types::CPos ad = x[q[3]].getCenter() - x[q[0]].getCenter();

    const defaulttype::Rigid3Types::CPos p1 = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));
    const defaulttype::Rigid3Types::CPos p2 = (ac.cross(ad)) * (pressure / static_cast<Real>(6.0));

    const defaulttype::Rigid3Types::CPos p = p1 + p2;

    f[q[0]].getVCenter() += p;
    f[q[1]].getVCenter() += p1;
    f[q[2]].getVCenter() += p;
    f[q[3]].getVCenter() += p2;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::verifyDerivative(VecDeriv& /*v_plus*/, VecDeriv& /*v*/, VecVec3DerivValues& /*DVval*/, VecVec3DerivIndices& /*DVind*/, const VecDeriv& /*Din*/)
{}

using namespace sofa::defaulttype;

void registerSurfacePressureForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Pressure applied on a generic surface (triangular or quadrangular).")
        .add<SurfacePressureForceField<Vec3Types> >()
        .add<SurfacePressureForceField<Rigid3Types> >());
}

template class SOFA_COMPONENT_MECHANICALLOAD_API SurfacePressureForceField<Vec3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API SurfacePressureForceField<Rigid3Types>;

} // namespace sofa::component::mechanicalload
