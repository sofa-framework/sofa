/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHBENDINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHBENDINGFORCEFIELD_INL

#include <SofaDeformable/ClothBendingForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>



namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template <class DataTypes>
ClothBendingForceField<DataTypes>::ClothBendingForceField():
    kb(initData(&kb,1.0,"bending","flexural rigidity for the all springs")),
    debug(initData(&debug,false,"debug","display all computed forces (slow)"))
{
}


template <class DataTypes>
ClothBendingForceField<DataTypes>::~ClothBendingForceField()
{
}


template<class DataTypes>
void ClothBendingForceField<DataTypes>::addSpringForce(Real& potentialEnergy,VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, 
                                                        VecDeriv& f2, const  VecCoord& p2, const  VecDeriv& v2, int i, const Spring& spring)
{
    int indiceP1 = spring.m1;
    int indiceP2 = spring.m2;
    Coord p1p2 = p2[indiceP2]-p1[indiceP1];
    Real length = p1p2.norm();
    Real restLength = spring.initpos;
    Mat& m = this->dfdx[i];

    if(length>1.0e-4 && length<restLength)
    {
        Coord forceDirection = p1p2/length;
        Real nonLinearFb = (Real)kb.getValue() * fb(length,restLength);
        Real linearFb = spring.ks*(length - restLength);
        Real fbJacobian = spring.ks;

        Deriv relativeVelocity = v2[indiceP2]-v1[indiceP1];
        Real elongationVelocity = dot(forceDirection,relativeVelocity);
        Real forceIntensity = spring.kd*elongationVelocity;

        if( nonLinearFb >= linearFb )
        { 
            forceIntensity += nonLinearFb;
            fbJacobian = (Real)kb.getValue() * dfb(length,restLength);
            if(debug.getValue())
                debug_colors.push_back(defaulttype::Vec4f(0,0,1,1));
            for( int j=0; j<N; ++j )
                for( int k=0; k<N; ++k )
                    m[j][k] = fbJacobian * forceDirection[j] * forceDirection[k];
        }
        else
        {
           if(debug.getValue())
                debug_colors.push_back(defaulttype::Vec4f(0,1,0,1));

           forceIntensity += linearFb;
           Real tgt = forceIntensity / length;
           for( int j=0; j<N; ++j )
           {
               for( int k=0; k<N; ++k )
                   m[j][k] = (fbJacobian - tgt)* forceDirection[j] * forceDirection[k];
               m[j][j] += tgt; 
           }
        }
        
        Deriv force = forceDirection*forceIntensity;

        f1[indiceP1]+=force;
        f2[indiceP2]-=force;

        if(debug.getValue())
        {
            debug_forces.push_back(p1[indiceP1]);
            debug_forces.push_back(p1[indiceP1]+force);
            debug_forces.push_back(p2[indiceP2]);
            debug_forces.push_back(p2[indiceP2]-force);
        }

        potentialEnergy += (length - restLength)*(length - restLength) * spring.ks / 2;

        if (this->maskInUse)
        {
            this->mstate1->forceMask.insertEntry(indiceP1);
            this->mstate2->forceMask.insertEntry(indiceP2);
        }

        
    }
    else // null length, no force and no stiffness
    {
        this->dfdx[i].clear();
    }
}

template<class DataTypes>
void ClothBendingForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, 
                                                 const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{
    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    if(debug.getValue())
    {
        debug_forces.clear();
        debug_colors.clear();
    }

    const helper::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        //
        this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);

    }
    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void ClothBendingForceField<DataTypes>::addSpring( unsigned a, unsigned b, std::set<IndexPair>& springSet )
{
    IndexPair ab(a<b?a:b, a<b?b:a);
    if (springSet.find(ab) != springSet.end()) return;
    springSet.insert(ab);
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    Spring spring(a,b,s,d,l); 
    this->SpringForceField<DataTypes>::addSpring(spring);
}


template<class DataTypes>
bool ClothBendingForceField<DataTypes>::findAlignedEdges(const unsigned index1, const unsigned index2, unsigned& index3)
{
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();

    Real angle;
    bool finded = false;

    sofa::core::topology::BaseMeshTopology::QuadsAroundVertex quadsAroundVertex = topology->getQuadsAroundVertex(index2);
    sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex trianglesAroundVertex = topology->getTrianglesAroundVertex(index2);

    //quads
    for(unsigned i=0; i<quadsAroundVertex.size(); i++)
    {
        const sofa::core::topology::BaseMeshTopology::Quad& face = quads[quadsAroundVertex[i]];
        for(unsigned j=0; j<face.size(); j++)
        {
            if(face[j]!=index1 && face[j]!=index2)
            {
                //sout<<"testing alignment "<<index1<<" "<<index2<<" "<< face[j]<<" ... ";
                Coord e1 = x[index2]  - x[index1];
                Coord e2 = x[face[j]] - x[index2];
                Real tmpAngle = 1-e1.normalized()*e2.normalized();
                if ((std::abs(tmpAngle) <  0.05) && (!finded || std::abs(tmpAngle)<angle)) // 0.05 represent an angle under 18 degrees
                {
                    index3 = face[j];
                    angle = std::abs(tmpAngle);
                    finded = true;
                }
            }
        }
    }

    //triangles
    for(unsigned i=0; i<trianglesAroundVertex.size(); i++)
    {
        const sofa::core::topology::BaseMeshTopology::Triangle& face = triangles[trianglesAroundVertex[i]];
        for(unsigned j=0; j<face.size(); j++)
        {
            if(face[j]!=index1 && face[j]!=index2)
            {
                //sout<<"testing alignment "<<index1<<" "<<index2<<" "<< face[j]<<" ... ";
                Coord e1 = x[index2]  - x[index1];
                Coord e2 = x[face[j]] - x[index2];
                Real tmpAngle = 1-e1.normalized()*e2.normalized();
                if ((std::abs(tmpAngle) <  0.05) && (!finded || std::abs(tmpAngle)<angle)) // 0.05 represent an angle under 18 degrees
                {
                    index3 = face[j];
                    angle = std::abs(tmpAngle);
                    finded = true;
                }
            }
        }
    }

    return finded;
}

template<class DataTypes>
void ClothBendingForceField<DataTypes>::init()
{
    // 

    std::map< IndexPair, IndexPair > edgeMap;
    std::set< IndexPair > springSet;
    unsigned index2;
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    assert( topology );
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();

    for(unsigned index1=0; index1<(unsigned)topology->getNbPoints(); ++index1)
    {
        //quads
        sofa::core::topology::BaseMeshTopology::QuadsAroundVertex quadsAroundVertex = topology->getQuadsAroundVertex(index1);
        for( unsigned i= 0; i<quadsAroundVertex.size(); ++i )
        {
            const sofa::core::topology::BaseMeshTopology::Quad& face = quads[quadsAroundVertex[i]];
            for(unsigned j=0; j<face.size();j++)
            {
                if(face[j]!=index1 && findAlignedEdges(index1, face[j], index2))
                {
                    addSpring(index1, index2, springSet);
                }
            }
        }
        
        //triangles
        sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex trianglesAroundVertex = topology->getTrianglesAroundVertex(index1);
        for( unsigned i= 0; i<trianglesAroundVertex.size(); ++i )
        {
            const sofa::core::topology::BaseMeshTopology::Triangle& face = triangles[trianglesAroundVertex[i]];
            for(unsigned j=0; j<face.size();j++)
            {
                if(face[j]!=index1 && findAlignedEdges(index1, face[j], index2))
                {
                    addSpring(index1, index2, springSet);
                }
            }
        }
    }
    // init the parent class
    StiffSpringForceField<DataTypes>::init();
}

template<class DataTypes>
void ClothBendingForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(debug.getValue())
    {
        defaulttype::Vec4f color(0,0,1,1);
        std::vector<Vector3> force(2);
        for(unsigned i=0; i<debug_forces.size()/4; i++)
        {
            force[0] = debug_forces[i*4];
            force[1] = debug_forces[i*4+1];
            vparams->drawTool()->drawLines(force, 1, debug_colors[i]);
            force[0] = debug_forces[i*4+2];
            force[1] = debug_forces[i*4+3];
            vparams->drawTool()->drawLines(force, 1, debug_colors[i]);
        }
    }
    this->StiffSpringForceField::draw(vparams);
}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL */
