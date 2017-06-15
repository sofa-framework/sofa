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
#ifndef SOFA_HEMLStVKFORCEFIELD_H
#define SOFA_HEMLStVKFORCEFIELD_H

#include <Flexible/config.h>

#include <sofa/core/behavior/ForceField.h>


#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>



namespace sofa
{
namespace component
{
namespace forcefield
{




/// HEML implementation of St Venant-Kirchhoff material (based on the square lengths of tetrahedron edges)
/// F. Goulette, Z.-W. Chen. Fast computation of soft tissue deformations in real-time simulation with Hyper-Elastic Mass Links. Computer Methods in Applied Mechanics and Engineering, 2015
///
/// Closely related to:
/// H. Delingette, Biquadratic and Quadratic Springs for Modeling St Venant Kirchhoff Materials, ISBMS 2008
/// Ryo Kikuuwe, Hiroaki Tabuchi, and Motoji Yamamoto, An Edge-Based Computationally Efficient Formulation of Saint Venant-Kirchhoff Tetrahedral Finite Elements, ACM ToG 2009
///
/// @author Matthieu Nesme
/// @date 2016
///
/// TODO have a common API for every HEML materials
/// TODO volumes could be computed from a specific GPsampler to better fit Flexible
///
template <class DataTypes>
class HEMLStVKForceField : public core::behavior::ForceField<DataTypes>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE(HEMLStVKForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename Inherit1::Real Real;
    typedef defaulttype::StdVectorTypes<defaulttype::Vec<3,Real>,defaulttype::Vec<3,Real>,Real> DataTypes3;

    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<6,6,Real> Mat66;
    typedef typename DataTypes3::Deriv Vec3;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef typename DataTypes3::VecCoord VecCoord3;

    typedef typename Inherit1::VecCoord VecCoord;
    typedef typename Inherit1::DataVecCoord DataVecCoord;
    typedef typename Inherit1::VecDeriv VecDeriv;
    typedef typename Inherit1::DataVecDeriv DataVecDeriv;


    /** @name  Material parameters */
    //@{
    Data<Real> d_youngModulus;
    Data<Real> d_poissonRatio;
    //@}



    virtual void bwdInit()
    {
        Inherit1::bwdInit();
        reinit();
    }

    virtual void reinit()
    {
        Inherit1::reinit();

        // lame coef
        Real lambda = d_youngModulus.getValue()*d_poissonRatio.getValue()/((1 + d_poissonRatio.getValue())*(1 - 2*d_poissonRatio.getValue()));
        Real mu = d_youngModulus.getValue()/(2*(1 + d_poissonRatio.getValue()));

        static const Mat33 D[6] = { Mat33( Vec3(1,0.5,0.5), Vec3(0.5,0,0), Vec3(0.5,0,0)),
                                    Mat33( Vec3(0,0.5,0), Vec3(0.5,1,0.5), Vec3(0,0.5,0)),
                                    Mat33( Vec3(0,0,0.5), Vec3(0,0,0.5), Vec3(0.5,0.5,1)),
                                    Mat33( Vec3(0,-0.5,0), Vec3(-0.5,0,0), Vec3(0,0,0)),
                                    Mat33( Vec3(0,0,-0.5), Vec3(0,0,0), Vec3(-0.5,0,0)),
                                    Mat33( Vec3(0,0,0), Vec3(0,0,-0.5), Vec3(0,-0.5,0)), };
                // warning indices of edges are not the same in sofa topology and in the HEML article  (l5 and l6 are inverted)


        topology::MeshTopology* topology = this->getContext()->template get<topology::MeshTopology>( core::objectmodel::BaseContext::SearchUp );
        if( !topology ) serr<<"No Topology found ! "<<sendl;


        // big mess to get positions...
        const VecCoord3* _points;
        Data<VecCoord3>* datapoints = static_cast<Data<VecCoord3>*>(topology->findData("position"));
        if( !datapoints || datapoints->getValue().empty() )
            _points = static_cast<const VecCoord3*>(topology->getContext()->getState()->baseRead(core::ConstVecCoordId::position())->getValueVoidPtr());
        else
            _points = &datapoints->getValue();
        const VecCoord3& pos = *_points;



        m_K.resize( this->getMState()->getMatrixSize(), this->getMState()->getMatrixSize() );

        const core::topology::BaseMeshTopology::SeqTetrahedra& tetras = topology->getTetrahedra();
        const size_t nbtetras = tetras.size();

        for( size_t t=0 ; t<nbtetras ; ++t )
        {
            const core::topology::BaseMeshTopology::Tetra& tetra = tetras[t];

            const Vec3 edge0 = pos[tetra[1]]-pos[tetra[0]];
            const Vec3 edge1 = pos[tetra[2]]-pos[tetra[0]];
            const Vec3 edge2 = pos[tetra[3]]-pos[tetra[0]];

            // TODO handle unstable configurations

            const Mat33 V( Vec3( edge0[0], edge1[0], edge2[0] ), Vec3( edge0[1], edge1[1], edge2[1] ), Vec3( edge0[2], edge1[2], edge2[2] ) );
            Mat33 Vinv; Vinv.invert( V );

            Mat33 C[6];
            for( int i=0 ; i<6 ; ++i )
                C[i] = Vinv.multTranspose( D[i] * Vinv );


            Vec6 vtr;
            defaulttype::Mat<6,6,Real> Mtr;

            for( int i=0 ; i<6 ; ++i )
            {
                vtr[i] = trace(C[i]);
                Mtr[i][i] = trace( C[i]*C[i] ); // TODO no need for computing the entire matrice product
                for( int j=i+1 ; j<6 ; ++j )
                    Mtr[i][j] = Mtr[j][i] = trace( C[i]*C[j] ); // TODO no need for computing the entire matrice product
            }

            Mat66 M = lambda/8.0*( defaulttype::tensorProduct( vtr, vtr ) ) + mu/4.0*Mtr;
            M *= std::abs( dot(edge0,cross(edge1,edge2)) / 6.0 ); // vol   TODO get volume from a Gauss point sampler

            const core::topology::BaseMeshTopology::EdgesInTetrahedron& edges = topology->getEdgesInTetrahedron(t);
            for( int i=0 ; i<6 ; ++i )
                for( int j=0 ; j<6 ; ++j )
                    m_K.add( edges[i], edges[j], -2*M[i][j] ); // K = -2 M
        }

        m_K.compress();
    }


    // W = (l-l0)^T M (l-l0)
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& _x) const
    {
        const VecCoord& x = _x.getValue();
        const VecCoord& x0 = this->getMState()->read(core::ConstVecCoordId::restPosition())->getValue();

        // berk
        VecCoord dx(x.size()), f(x.size());
        for( size_t i=0;i<x.size();++i)
            dx[i]=x[i]-x0[i];

        m_K.addMult( f, dx, -0.5 );  // K = -2 M

        SReal e = 0;
        for( size_t i=0;i<x.size();++i)
            e += dx[i] * f[i];

        return e;
    }


    // f += -2 (l-l0)^T M == K (l-l0)
    virtual void addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& /*_v*/)
    {
        VecDeriv& f = *_f.beginEdit();
        const VecCoord& x = _x.getValue();
        const VecCoord& x0 = this->getMState()->read(core::ConstVecCoordId::restPosition())->getValue();

        // berk
        VecCoord dx( x.size() );
        for( size_t i=0 ; i<x.size() ; ++i )
            dx[i] = x[i] - x0[i];


        m_K.addMult( f, dx );

        _f.endEdit();
    }


    // df += -2 M dx == K dx
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    {
        m_K.addMult( df, dx, mparams->kFactor()  );
    }


    // K = -2 M
    void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
    {
        m_K.addToBaseMatrix( matrix, kFact, offset );
    }



protected:
    HEMLStVKForceField(core::behavior::MechanicalState<DataTypes> *mm = NULL)
        : Inherit1(mm)
        , d_youngModulus(initData(&d_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , d_poissonRatio(initData(&d_poissonRatio,(Real)0,"poissonRatio","Poisson Ratio ]-1,0.5["))
    {}

    virtual ~HEMLStVKForceField() {}


    linearsolver::EigenSparseMatrix<DataTypes,DataTypes> m_K; ///< assembled stiffness matrix (per edge)

};








}
}
}

#endif


