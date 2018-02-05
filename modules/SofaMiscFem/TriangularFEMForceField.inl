/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "TriangularFEMForceField.h"

#include <sofa/core/visual/VisualParams.h>
#include <SofaOpenglVisual/OglColorMap.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/gl.h>

#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/helper/system/thread/debug.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <algorithm>
#include <limits>

#ifdef DEBUG_TRIANGLEFEM
    #define DEBUG_TRIANGLEFEM_MSG true
#else
    #define DEBUG_TRIANGLEFEM_MSG false
#endif



namespace sofa
{

namespace component
{

namespace forcefield
{

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void TriangularFEMForceField<DataTypes>::TRQSTriangleHandler::applyCreateFunction(unsigned int triangleIndex, TriangleInformation &, const core::topology::BaseMeshTopology::Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];

        switch(ff->method)
        {
        case SMALL :
            ff->initSmall(triangleIndex,a,b,c);
            ff->computeMaterialStiffness(triangleIndex,a,b,c);
            break;

        case LARGE :
            ff->initLarge(triangleIndex,a,b,c);
            ff->computeMaterialStiffness(triangleIndex,a,b,c);
            break;
        }
    }
}


// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
TriangularFEMForceField<DataTypes>::TriangularFEMForceField()
    : triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
    , vertexInfo(initData(&vertexInfo, "vertexInfo", "Internal point data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , _topology(NULL)
    , method(LARGE)
    , f_method(initData(&f_method,std::string("large"),"method","large: large displacements, small: small displacements"))
//, f_poisson(initData(&f_poisson,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
//, f_young(initData(&f_young,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_poisson(initData(&f_poisson,helper::vector<Real>(1,static_cast<Real>(0.45)),"poissonRatio","Poisson ratio in Hooke's law (vector)"))
    , f_young(initData(&f_young,helper::vector<Real>(1,static_cast<Real>(1000.0)),"youngModulus","Young modulus in Hooke's law (vector)"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , m_rotatedInitialElements(initData(&m_rotatedInitialElements,"rotatedInitialElements","Flag activating rendering of stress directions within each triangle"))
    , m_initialTransformation(initData(&m_initialTransformation,"initialTransformation","Flag activating rendering of stress directions within each triangle"))
    , f_fracturable(initData(&f_fracturable,false,"fracturable","the forcefield computes the next fracturable Edge"))
    , hosfordExponant(initData(&hosfordExponant, (Real)1.0, "hosfordExponant","Exponant in the Hosford yield criteria"))
    , criteriaValue(initData(&criteriaValue, (Real)1e15, "criteriaValue","Fracturable threshold used to draw fracturable triangles"))
    , showStressValue(initData(&showStressValue,false,"showStressValue","Flag activating rendering of stress values as a color in each triangle"))
    , showStressVector(initData(&showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
    , showFracturableTriangles(initData(&showFracturableTriangles,false,"showFracturableTriangles","Flag activating rendering of triangles to fracture"))
    , f_computePrincipalStress(initData(&f_computePrincipalStress,false,"computePrincipalStress","Compute principal stress for each triangle"))
#ifdef PLOT_CURVE
    , elementID( initData(&elementID, (Real)0, "id","element id to follow for fracture criteria") )
    , f_graphStress( initData(&f_graphStress,"graphMaxStress","Graph of max stress corresponding to the element id") )
    , f_graphCriteria( initData(&f_graphCriteria,"graphCriteria","Graph of the fracture criteria corresponding to the element id") )
    , f_graphOrientation( initData(&f_graphOrientation,"graphOrientation","Graph of the orientation of the principal stress direction corresponding to the element id"))
#endif
{
    _anisotropicMaterial = false;
    triangleHandler = new TRQSTriangleHandler(this, &triangleInfo);
#ifdef PLOT_CURVE
    f_graphStress.setWidget("graph");
    f_graphCriteria.setWidget("graph");
    f_graphOrientation.setWidget("graph");
#endif

    f_poisson.setRequired(true);
    f_young.setRequired(true);
}


template <class DataTypes>
TriangularFEMForceField<DataTypes>::~TriangularFEMForceField()
{
    if(triangleHandler) delete triangleHandler;
}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::init()
{
    this->Inherited::init();

#ifdef PLOT_CURVE
    allGraphStress.clear();
    allGraphCriteria.clear();
    allGraphOrientation.clear();

    f_graphStress.beginEdit()->clear();
    f_graphStress.endEdit();
    f_graphCriteria.beginEdit()->clear();
    f_graphCriteria.endEdit();
    f_graphOrientation.beginEdit()->clear();
    f_graphOrientation.endEdit();
#endif

    _topology = this->getContext()->getMeshTopology();

    // Create specific handler for TriangleData
    triangleInfo.createTopologicalEngine(_topology, triangleHandler);
    triangleInfo.registerTopologicalData();

    edgeInfo.createTopologicalEngine(_topology);
    edgeInfo.registerTopologicalData();

    vertexInfo.createTopologicalEngine(_topology);
    vertexInfo.registerTopologicalData();


    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    if (_topology->getNbTriangles()==0)
    {
        serr << "ERROR(TriangularFEMForceField): object must have a Triangular Set Topology."<<sendl;
        return;
    }

    lastFracturedEdgeIndex = -1;

    reinit();
}


// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initSmall(int i, Index&a, Index&b, Index&c)
{

    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering initSmall" ;

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    TriangleInformation *tinfo = &triangleInf[i];

    tinfo->initialTransformation.identity();

    if (m_rotatedInitialElements.isSet())
        tinfo->rotatedInitialElements = m_rotatedInitialElements.getValue()[i];
    else
    {
        const  VecCoord& initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());

        tinfo->rotatedInitialElements[0] = (initialPoints)[a] - (initialPoints)[a]; // always (0,0,0)
        tinfo->rotatedInitialElements[1] = (initialPoints)[b] - (initialPoints)[a];
        tinfo->rotatedInitialElements[2] = (initialPoints)[c] - (initialPoints)[a];
    }

    computeStrainDisplacement(tinfo->strainDisplacementMatrix, i, tinfo->rotatedInitialElements[0], tinfo->rotatedInitialElements[1], tinfo->rotatedInitialElements[2]);

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initLarge(int i, Index&a, Index&b, Index&c)
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering initLarge" ;

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    msg_error_when((unsigned int)i >= triangleInf.size())
            << "Try to access an element which indices bigger than the size of the vector: i=" << i << " and size=" << triangleInf.size() ;

    TriangleInformation *tinfo = &triangleInf[i];

    if (m_initialTransformation.isSet() && m_rotatedInitialElements.isSet())
    {
        Transformation R_0_1;
        R_0_1 = m_initialTransformation.getValue()[i];
        tinfo->initialTransformation = R_0_1;
        tinfo->rotatedInitialElements = m_rotatedInitialElements.getValue()[i];
    }
    else
    {
        // Rotation matrix (initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1;

         VecCoord initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());

        computeRotationLarge( R_0_1, (initialPoints), a, b, c );

        tinfo->initialTransformation = R_0_1;

        if ( a >= (initialPoints).size() || b >= (initialPoints).size() || c >= (initialPoints).size() )
        {
            std::stringstream tmp;
            tmp << "Try to access an element which indices bigger than the size of the vector: a=" <<a <<
                    " b=" << b << " and c=" << c << " and size=" << (initialPoints).size() << msgendl;

            //reset initialPoints in case of a new pointer of the initial points of the mechanical state
            initialPoints = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

            tmp << "Now it's: a=" <<a <<
                    " b=" << b << " and c=" << c << " and size=" << (initialPoints).size() ;

            msg_error() << tmp.str() ;
        }


        tinfo->rotatedInitialElements[0] = R_0_1 * ((initialPoints)[a] - (initialPoints)[a]);
        tinfo->rotatedInitialElements[1] = R_0_1 * ((initialPoints)[b] - (initialPoints)[a]);
        tinfo->rotatedInitialElements[2] = R_0_1 * ((initialPoints)[c] - (initialPoints)[a]);
    }

    computeStrainDisplacement(tinfo->strainDisplacementMatrix, i, tinfo->rotatedInitialElements[0], tinfo->rotatedInitialElements[1], tinfo->rotatedInitialElements[2]);

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    /// prepare to store info in the triangle array
    triangleInf.resize(_topology->getNbTriangles());

    /// prepare to store info in the edge array
    edgeInf.resize(_topology->getNbEdges());


    unsigned int nbPoints = _topology->getNbPoints();
    helper::vector<VertexInformation>& vi = *(vertexInfo.beginEdit());
    vi.resize(nbPoints);
    vertexInfo.endEdit();


    for (int i=0; i<_topology->getNbTriangles(); ++i)
    {
        triangleHandler->applyCreateFunction(i, triangleInf[i],  _topology->getTriangle(i),  (const sofa::helper::vector< unsigned int > )0, (const sofa::helper::vector< double >)0);
    }

    edgeInfo.endEdit();
    triangleInfo.endEdit();

#ifdef PLOT_CURVE
    std::map<std::string, sofa::helper::vector<double> > &stress = *(f_graphStress.beginEdit());
    stress.clear();
    if (allGraphStress.size() > elementID.getValue())
        stress = allGraphStress[elementID.getValue()];
    f_graphStress.endEdit();

    std::map<std::string, sofa::helper::vector<double> > &criteria = *(f_graphCriteria.beginEdit());
    criteria.clear();
    if (allGraphCriteria.size() > elementID.getValue())
        criteria = allGraphCriteria[elementID.getValue()];
    f_graphCriteria.endEdit();

    std::map<std::string, sofa::helper::vector<double> > &orientation = *(f_graphOrientation.beginEdit());
    orientation.clear();
    if (allGraphOrientation.size() > elementID.getValue())
        orientation = allGraphOrientation[elementID.getValue()];
    f_graphOrientation.endEdit();
#endif
}





// --------------------------------------------------------------------------------------
// --- Get/Set methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
SReal TriangularFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
    serr<<"TriangularFEMForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getRotation(Transformation& R, unsigned int nodeIdx)
{
    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    int numNeiTri=_topology->getTrianglesAroundVertex(nodeIdx).size();
    Transformation r;
    for(int i=0; i<numNeiTri; i++)
    {
        int triIdx=_topology->getTrianglesAroundVertex(nodeIdx)[i];
        TriangleInformation *tinfo = &triangleInf[triIdx];
        Transformation r01,r21;
        r01=tinfo->initialTransformation;
        r21=tinfo->rotation*r01;
        r+=r21;
    }
    R=r/static_cast<Real>(numNeiTri);

    //orthogonalization
    Coord ex,ey,ez;
    for(int i=0; i<3; i++)
    {
        ex[i]=R[0][i];
        ey[i]=R[1][i];
    }
    ex.normalize();
    ey.normalize();

    ez=cross(ex,ey);
    ez.normalize();

    ey=cross(ez,ex);
    ey.normalize();

    for(int i=0; i<3; i++)
    {
        R[0][i]=ex[i];
        R[1][i]=ey[i];
        R[2][i]=ez[i];
    }
    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getRotations()
{
    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    helper::vector<VertexInformation>& vertexInf = *(vertexInfo.beginEdit());
    int numPoint=_topology->getNbPoints();
    int numTri=_topology->getNbTriangles();

    //reset the rotation matrix
    for(int i=0; i<numPoint; i++)
    {
        VertexInformation *vinfo= &vertexInf[i];
        vinfo->rotation.clear();
    }

    //add the rotation matrix
    for(int i=0; i<numTri; i++)
    {
        TriangleInformation *tinfo = &triangleInf[i];
        Transformation r01,r21;
        r01=tinfo->initialTransformation;
        r21=tinfo->rotation*r01;

        for(int j=0; j<3; j++)
        {
            int idx=_topology->getTriangle(i)[j];
            VertexInformation *vinfo= &vertexInf[idx];
            vinfo->rotation+=r21;
        }
    }

    //averaging the rotation matrix
    for(int i=0; i<numPoint; i++)
    {
        VertexInformation *vinfo=&vertexInf[i];
        int numNeiTri=_topology->getTrianglesAroundVertex(i).size();
        vinfo->rotation/=static_cast<Real>(numNeiTri);

        //orthogonalization
        Coord ex,ey,ez;
        for(int i=0; i<3; i++)
        {
            ex[i]=vinfo->rotation[0][i];
            ey[i]=vinfo->rotation[1][i];
        }
        ex.normalize();
        ey.normalize();

        ez=cross(ex,ey);
        ez.normalize();

        ey=cross(ez,ex);
        ey.normalize();

        for(int i=0; i<3; i++)
        {
            vinfo->rotation[0][i]=ex[i];
            vinfo->rotation[1][i]=ey[i];
            vinfo->rotation[2][i]=ez[i];
        }
    }
    triangleInfo.endEdit();
    vertexInfo.endEdit();
}


// --------------------------------------------------------------------------------------
// --- Get Fracture Criteria
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::getFractureCriteria(int elementIndex, Deriv& direction, Real& value)
{
    //TODO(dmarchal 2017-05-03) Who wrote this todo ? When will you fix this ? In one year I remove this one.
    /// @todo evaluate the criteria on the current position instead of relying on the computations during the force evaluation (based on the previous position)
    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    if ((unsigned)elementIndex < triangleInf.size())
    {
        computePrincipalStress(elementIndex, triangleInf[elementIndex].stress);
        direction = triangleInf[elementIndex].principalStressDirection;
        value = fabs(triangleInf[elementIndex].maxStress);
        if (value < 0)
        {
            direction.clear();
            value = 0;
        }
        triangleInfo.endEdit();
    }
    else
    {
        direction.clear();
        value = 0;
    }
}

template<class DataTypes>
int TriangularFEMForceField<DataTypes>::getFracturedEdge()
{
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());

    if (f_fracturable.getValue())
    {
        int nbEdges = _topology->getNbEdges();

        for( int i=0; i<nbEdges; i++ )
        {
            if (edgeInf[i].fracturable)
            {
                return i;
            }
        }
    }

    edgeInfo.endEdit();

    return -1;
}








// --------------------------------------------------------------------------------------
// --- Computation methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c)
{

    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering in computeRotationLarge.";

    /// check if a, b and c are < size of p
    if (a >= p.size() || b >= p.size() || c >= p.size())
    {
        msg_error() <<  "Indices given in parameters are wrong>> a=" << a << " b=" << b << " and c=" << c <<
                " whereas the size of the vector p is " << p.size() ;
        return;
    }

    /// first vector on first edge
    /// second vector in the plane of the two first edges
    /// third vector orthogonal to first and second
    Coord edgex = p[b] - p[a];
    edgex.normalize();

    Coord edgey = p[c] - p[a];
    edgey.normalize();

    Coord edgez;
    edgez = cross(edgex, edgey);
    edgez.normalize();

    edgey = cross(edgez, edgex);
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];

    if ( r[0][0]!=r[0][0])
    {
        msg_info() << "computeRotationLarge::edgex " << edgex << msgendl
                   << "computeRotationLarge::edgey " << edgey << msgendl
                   << "computeRotationLarge::edgez " << edgez << msgendl
                   << "computeRotationLarge::pa " << p[a] << msgendl
                   << "computeRotationLarge::pb " << p[b] << msgendl
                   << "computeRotationLarge::pc " <<  p[c] << msgendl;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ---	Compute displacement vector D as the difference between current current position 'p' and initial position
// ---------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeDisplacementSmall(Displacement &D, Index elementIndex, const VecCoord &p)
{
    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    //Coord deforme_a = Coord(0,0,0);
    Coord deforme_b = p[b]-p[a];
    Coord deforme_c = p[c]-p[a];

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    D[0] = 0;
    D[1] = 0;
    D[2] = triangleInf[elementIndex].rotatedInitialElements[1][0] - deforme_b[0];
    D[3] = triangleInf[elementIndex].rotatedInitialElements[1][1] - deforme_b[1];
    D[4] = triangleInf[elementIndex].rotatedInitialElements[2][0] - deforme_c[0];
    D[5] = triangleInf[elementIndex].rotatedInitialElements[2][1] - deforme_c[1];
    triangleInfo.endEdit();
}

// -------------------------------------------------------------------------------------------------------------
// --- Compute displacement vector D as the difference between current current position 'p' and initial position
// --- expressed in the co-rotational frame of reference
// -------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeDisplacementLarge(Displacement &D, Index elementIndex, const Transformation &R_0_2, const VecCoord &p)
{
    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    // positions of the deformed and displaced triangle in its frame
    Coord deforme_b = R_0_2 * (p[b]-p[a]);
    Coord deforme_c = R_0_2 * (p[c]-p[a]);

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    // displacements
    D[0] = 0;
    D[1] = 0;
    D[2] = triangleInf[elementIndex].rotatedInitialElements[1][0] - deforme_b[0];
    D[3] = 0;
    D[4] = triangleInf[elementIndex].rotatedInitialElements[2][0] - deforme_c[0];
    D[5] = triangleInf[elementIndex].rotatedInitialElements[2][1] - deforme_c[1];

    if ( D[2] != D[2] || D[4] != D[4] || D[5] != D[5])
    {
        msg_info() << "computeDisplacementLarge :: deforme_b = " <<  deforme_b << msgendl
                   << "computeDisplacementLarge :: deforme_c = " <<  deforme_c << msgendl
                   << "computeDisplacementLarge :: R_0_2 = " <<  R_0_2 << msgendl;
    }

    triangleInfo.endEdit();
}

// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix where (a, b, c) are the coordinates of the 3 nodes of a triangle
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStrainDisplacement(StrainDisplacement &J, Index elementIndex, Coord a, Coord b, Coord c )
{
    Real determinant;
    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    if (method == SMALL)
    {
        Coord ab_cross_ac = cross(b-a, c-a);
        determinant = ab_cross_ac.norm();
        triangleInf[elementIndex].area = determinant*0.5f;

        Real x13 = (a[0]-c[0]) / determinant;
        Real x21 = (b[0]-a[0]) / determinant;
        Real x32 = (c[0]-b[0]) / determinant;
        Real y12 = (a[1]-b[1]) / determinant;
        Real y23 = (b[1]-c[1]) / determinant;
        Real y31 = (c[1]-a[1]) / determinant;

        J[0][0] = y23;
        J[0][1] = 0;
        J[0][2] = x32;

        J[1][0] = 0;
        J[1][1] = x32;
        J[1][2] = y23;

        J[2][0] = y31;
        J[2][1] = 0;
        J[2][2] = x13;

        J[3][0] = 0;
        J[3][1] = x13;
        J[3][2] = y31;

        J[4][0] = y12;
        J[4][1] = 0;
        J[4][2] = x21;

        J[5][0] = 0;
        J[5][1] = x21;
        J[5][2] = y12;
    }
    else
    {
        determinant = b[0] * c[1];
        triangleInf[elementIndex].area = determinant*0.5f;

        Real x13 = -c[0] / determinant; // since a=(0,0)
        Real x21 = b[0] / determinant; // since a=(0,0)
        Real x32 = (c[0]-b[0]) / determinant;
        Real y12 = 0;	// since a=(0,0) and b[1] = 0
        Real y23 = -c[1] / determinant; // since a=(0,0) and b[1] = 0
        Real y31 = c[1] / determinant; // since a=(0,0)

        J[0][0] = y23; // -cy   / det
        J[0][1] = 0;   // 0
        J[0][2] = x32; // cx-bx / det

        J[1][0] = 0;   // 0
        J[1][1] = x32; // cx-bx / det
        J[1][2] = y23; // -cy   / det

        J[2][0] = y31; // cy    / det
        J[2][1] = 0;   // 0
        J[2][2] = x13; // -cx   / det

        J[3][0] = 0;   // 0
        J[3][1] = x13; // -cx   / det
        J[3][2] = y31; // cy    / det

        J[4][0] = y12; // 0
        J[4][1] = 0;   // 0
        J[4][2] = x21; // bx    / det

        J[5][0] = 0;   // 0
        J[5][1] = x21; // bx    / det
        J[5][2] = y12; // 0
    }
    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------------------------
// --- Stiffness = K = J*D*Jt
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStiffness(StrainDisplacement &J, Stiffness &K, MaterialStiffness &D)
{
    defaulttype::Mat<3,6,Real> Jt;
    Jt.transpose(J);
    K=J*D*Jt;
}

// --------------------------------------------------------------------------------------------------------
// --- Strain = StrainDisplacement * Displacement = JtD = Bd
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStrain(defaulttype::Vec<3,Real> &strain, const StrainDisplacement &J, const Displacement &D)
{
    defaulttype::Mat<3,6,Real> Jt;
    Jt.transpose(J);

    if (_anisotropicMaterial || method == SMALL)
    {
        strain = Jt * D;
    }
    else
    {
        strain[0] = Jt[0][0] * D[0] + /* Jt[0][1] * Depl[1] + */ Jt[0][2] * D[2] /* + Jt[0][3] * Depl[3] + Jt[0][4] * Depl[4] + Jt[0][5] * Depl[5] */ ;
        strain[1] = /* Jt[1][0] * Depl[0] + */ Jt[1][1] * D[1] + /* Jt[1][2] * Depl[2] + */ Jt[1][3] * D[3] + /* Jt[1][4] * Depl[4] + */ Jt[1][5] * D[5];
        strain[2] = Jt[2][0] * D[0] + Jt[2][1] * D[1] + Jt[2][2] * D[2] +	Jt[2][3] * D[3] + Jt[2][4] * D[4] /* + Jt[2][5] * Depl[5] */ ;
    }
}

// --------------------------------------------------------------------------------------------------------
// --- Stress = K * Strain = KJtD = KBd
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStress(defaulttype::Vec<3,Real> &stress, MaterialStiffness &K, defaulttype::Vec<3,Real> &strain)
{
    if (_anisotropicMaterial || method == SMALL)
    {
        stress = K * strain;
    }
    else
    {
        // Optimisations: The following values are 0 (per computeMaterialStiffnesses )
        // K[0][2]  K[1][2]  K[2][0] K[2][1]
        stress[0] = K[0][0] * strain[0] + K[0][1] * strain[1] + K[0][2] * strain[2];
        stress[1] = K[1][0] * strain[0] + K[1][1] * strain[1] + K[1][2] * strain[2];
        stress[2] = K[2][0] * strain[0] + K[2][1] * strain[1] + K[2][2] * strain[2];
    }
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum strain (strain = JtD = BD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computePrincipalStrain(Index elementIndex, defaulttype::Vec<3,Real> &strain )
{
    NEWMAT::SymmetricMatrix e(2);
    e = 0.0;

    NEWMAT::DiagonalMatrix D(2);
    D = 0.0;

    NEWMAT::Matrix V(2,2);
    V = 0.0;

    e(1,1) = strain[0];
    e(1,2) = strain[2];
    e(2,1) = strain[2];
    e(2,2) = strain[1];

    NEWMAT::Jacobi(e, D, V);

    Coord v((Real)V(1,1), (Real)V(2,1), 0.0);
    v.normalize();

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    triangleInf[elementIndex].maxStrain = (Real)D(1,1);

    triangleInf[elementIndex].principalStrainDirection = triangleInf[elementIndex].rotation * Coord(v[0], v[1], v[2]);
    triangleInf[elementIndex].principalStrainDirection *= triangleInf[elementIndex].maxStrain/100.0;

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum stress (stress = KJtD = KBD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computePrincipalStress(Index elementIndex, defaulttype::Vec<3,Real> &stress)
{
    NEWMAT::SymmetricMatrix e(2);
    e = 0.0;

    NEWMAT::DiagonalMatrix D(2);
    D = 0.0;

    NEWMAT::Matrix V(2,2);
    V = 0.0;

    //voigt notation to symmetric matrix
    e(1,1) = stress[0];
    e(1,2) = stress[2];
    e(2,1) = stress[2];
    e(2,2) = stress[1];

    //compute eigenvalues and eigenvectors
    NEWMAT::Jacobi(e, D, V);

    //get the index of the biggest eigenvalue in absolute value
    unsigned int biggestIndex = 0;
    if (fabs(D(1,1)) > fabs(D(2,2)))
        biggestIndex = 1;
    else
        biggestIndex = 2;

    //get the eigenvectors corresponding to the biggest eigenvalue
    //note : according to newmat doc => The elements of D are sorted in ascending order, The eigenvectors are returned as the columns of V
    Coord direction((Real)V(1,biggestIndex), (Real)V(2,biggestIndex), 0.0);
    direction.normalize();

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    //Hosford yield criterion
    //for plane stress : 1/2 * ( |S_1|^n + |S_2|^n) + 1/2 * |S_1 - S_2|^n = S_y^n
    //with S_i the principal stresses, n is a material-dependent exponent and S_y is the yield stress in uniaxial tension/compression
    double n = this->hosfordExponant.getValue();
    triangleInf[elementIndex].differenceToCriteria = (Real)
            pow(0.5 * (pow((double)fabs(D(1,1)), n) +  pow((double)fabs(D(2,2)), n) + pow((double)fabs(D(1,1) - D(2,2)),n)), 1.0/ n) - this->criteriaValue.getValue();

    //max stress is the highest eigenvalue
    triangleInf[elementIndex].maxStress = fabs((Real)D(biggestIndex,biggestIndex));

    //the principal stress direction is the eigenvector corresponding to the highest eigenvalue
    Coord principalStressDir = triangleInf[elementIndex].rotation * direction;//need to rotate to be in global frame instead of local
    principalStressDir *= triangleInf[elementIndex].maxStress/100.0;


    //make an average of the n1 and n2 last stress direction to smooth it and avoid discontinuities
    unsigned int n2 = 30;
    unsigned int n1 = 10;
    triangleInf[elementIndex].lastNStressDirection.push_back(principalStressDir);

    //remove useless data
    if (triangleInf[elementIndex].lastNStressDirection.size() > n2)
    {
        for ( unsigned int i = 0 ; i  < triangleInf[elementIndex].lastNStressDirection.size() - n2 ; i++)
            triangleInf[elementIndex].lastNStressDirection.erase(triangleInf[elementIndex].lastNStressDirection.begin()+i);
    }

    //make the average
    Coord averageVector2(0.0,0.0,0.0);
    Coord averageVector1(0.0,0.0,0.0);
    for (unsigned int i = 0 ; i < triangleInf[elementIndex].lastNStressDirection.size() ; i++)
    {
        averageVector2 = triangleInf[elementIndex].lastNStressDirection[i] + averageVector2;
        if (i == n1)
            averageVector1 = averageVector2 / n1;
    }
    if (triangleInf[elementIndex].lastNStressDirection.size())
        averageVector2 /=  triangleInf[elementIndex].lastNStressDirection.size();

    triangleInf[elementIndex].principalStressDirection = averageVector2;

#ifdef PLOT_CURVE
    Coord direction2((Real)V(1,D.Ncols() +1 - biggestIndex), (Real)V(2,D.Ncols() +1 - biggestIndex), 0.0);
    direction2.normalize();

    Coord principalStressDir2 = triangleInf[elementIndex].rotation * direction2;//need to rotate to be in global frame instead of local
    principalStressDir2 *= fabs((Real)D(D.Ncols() +1 - biggestIndex,D.Ncols() +1 - biggestIndex))/100.0;

    //compute an angle between the principal stress direction and the x-axis
    Real orientation2 = dot( averageVector2, Coord(1.0, 0.0, 0.0));
    Real orientation1 = dot( averageVector1, Coord(1.0, 0.0, 0.0));
    Real orientation0 = dot( principalStressDir, Coord(1.0, 0.0, 0.0));
    Real orientationSecond = dot( principalStressDir2, Coord(1.0, 0.0, 0.0));

    /* store the values which are plot*/
    if (allGraphStress.size() <= elementIndex)
        allGraphStress.resize(elementIndex+1);
    if (allGraphCriteria.size() <= elementIndex)
        allGraphCriteria.resize(elementIndex+1);
    if (allGraphOrientation.size() <= elementIndex)
        allGraphOrientation.resize(elementIndex+1);

    std::map<std::string, sofa::helper::vector<double> > &stressMap = allGraphStress[elementIndex];
    std::map<std::string, sofa::helper::vector<double> > &criteriaMap = allGraphCriteria[elementIndex];
    std::map<std::string, sofa::helper::vector<double> > &orientationMap = allGraphOrientation[elementIndex];

    stressMap["first stress eigenvalue"].push_back((double)(triangleInf[elementIndex].maxStress));
    stressMap["second stress eigenvalue"].push_back((double)(fabs(D(1,1))));

    criteriaMap["fracture criteria"].push_back((double)(triangleInf[elementIndex].differenceToCriteria));

    orientationMap["principal stress direction orientation with 30-average"].push_back((double)(acos(orientation2) * 180 / 3.14159265));
    orientationMap["principal stress direction orientation with 10-average"].push_back((double)(acos(orientation1) * 180 / 3.14159265));
    orientationMap["principal stress direction orientation with no-average"].push_back((double)(acos(orientation0) * 180 / 3.14159265));
    orientationMap["second stress direction orientation with no-average"].push_back((double)(acos(orientationSecond) * 180 / 3.14159265));


    //save values in graphs
    if (elementIndex == elementID.getValue())
    {
        std::map < std::string, sofa::helper::vector<double> >& graphStress = *f_graphStress.beginEdit();
        sofa::helper::vector<double>& graph_maxStress1 = graphStress["first stress eigenvalue"];
        graph_maxStress1.push_back((double)(triangleInf[elementIndex].maxStress));
        sofa::helper::vector<double>& graph_maxStress2 = graphStress["second stress eigenvalue"];
        graph_maxStress2.push_back((double)(fabs(D(1,1))));
        f_graphStress.endEdit();

        std::map < std::string, sofa::helper::vector<double> >& graphCriteria = *f_graphCriteria.beginEdit();
        sofa::helper::vector<double>& graph_criteria = graphCriteria["fracture criteria"];
        graph_criteria.push_back((double)(triangleInf[elementIndex].differenceToCriteria));
        f_graphCriteria.endEdit();

        std::map < std::string, sofa::helper::vector<double> >& graphOrientation = *f_graphOrientation.beginEdit();
        sofa::helper::vector<double>& graph_orientation2 = graphOrientation["principal stress direction orientation with 30-average"];
        graph_orientation2.push_back((double)(acos(orientation2) * 180 / 3.14159265));
        sofa::helper::vector<double>& graph_orientation1 = graphOrientation["principal stress direction orientation with 10-average"];
        graph_orientation1.push_back((double)(acos(orientation1) * 180 / 3.14159265));
        sofa::helper::vector<double>& graph_orientation0 = graphOrientation["principal stress direction orientation with no-average"];
        graph_orientation0.push_back((double)(acos(orientation0) * 180 / 3.14159265));
        sofa::helper::vector<double>& graph_orientationSecond = graphOrientation["second stress direction orientation with no-average"];
        graph_orientationSecond.push_back((double)(acos(orientationSecond) * 180 / 3.14159265));
        f_graphOrientation.endEdit();
    }
#endif

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// ---	Compute material stiffness
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index &/*a*/, Index &/*b*/, Index &/*c*/)
{
    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    const helper::vector<Real> & youngArray = f_young.getValue();
    const helper::vector<Real> & poissonArray = f_poisson.getValue();

    TriangleInformation *tinfo = &triangleInf[i];

    Real y = ((int)youngArray.size() > i ) ? youngArray[i] : youngArray[0] ;
    Real p = ((int)poissonArray.size() > i ) ? poissonArray[i] : poissonArray[0];

    tinfo->materialMatrix[0][0] = 1;
    tinfo->materialMatrix[0][1] = p;//poissonArray[i];//f_poisson.getValue();
    tinfo->materialMatrix[0][2] = 0;
    tinfo->materialMatrix[1][0] = p;//poissonArray[i];//f_poisson.getValue();
    tinfo->materialMatrix[1][1] = 1;
    tinfo->materialMatrix[1][2] = 0;
    tinfo->materialMatrix[2][0] = 0;
    tinfo->materialMatrix[2][1] = 0;
    tinfo->materialMatrix[2][2] = (1.0f - p) * 0.5f;//poissonArray[i]);

    tinfo->materialMatrix *= (y / (1.0f - p * p));

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// ---	Compute F = J * stress;
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeForce(Displacement &F, Index elementIndex, const VecCoord &p)
{
    //	sofa::helper::system::thread::Trace::print(1, "Hello from computeForce()\n");

    Displacement D;
    StrainDisplacement J;
    Stiffness K;
    defaulttype::Vec<3,Real> strain;
    defaulttype::Vec<3,Real> stress;
    Transformation R_0_2, R_2_0;

    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    if (method == SMALL)
    {
        // classic linear elastic method
        computeDisplacementSmall(D, elementIndex, p);
        computeStrainDisplacement(J, elementIndex, Coord(0,0,0), (p[b]-p[a]), (p[c]-p[a]));
        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[elementIndex].materialMatrix, strain);
        F = J * stress * triangleInf[elementIndex].area;

        // store newly computed values for next time
        triangleInf[elementIndex].strainDisplacementMatrix = J;
        triangleInf[elementIndex].strain = strain;
        triangleInf[elementIndex].stress = stress;
    }
    else
    {
        // co-rotational method
        // first, compute rotation matrix into co-rotational frame
        computeRotationLarge( R_0_2, p, a, b, c);

        // then compute displacement in this frame
        computeDisplacementLarge(D, elementIndex, R_0_2, p);

        // and compute postions of a, b, c in the co-rotational frame
        Coord A = Coord(0, 0, 0);
        Coord B = R_0_2 * (p[b]-p[a]);
        Coord C = R_0_2 * (p[c]-p[a]);

        if (_anisotropicMaterial)
            computeStrainDisplacement(J, elementIndex, A, B, C);
        else
            J = triangleInf[elementIndex].strainDisplacementMatrix;
        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[elementIndex].materialMatrix, strain);
        computeStiffness(J,K,triangleInf[elementIndex].materialMatrix);

        // Compute F = J * stress;
        // Optimisations: The following values are 0 (per computeStrainDisplacement )
        // J[0][1] J[1][0] J[2][1] J[3][0] J[4][0] J[4][1] J[5][0] J[5][2]

        F[0] = J[0][0] * stress[0] + /* J[0][1] * KJtD[1] + */ J[0][2] * stress[2];
        F[1] = /* J[1][0] * KJtD[0] + */ J[1][1] * stress[1] + J[1][2] * stress[2];
        F[2] = J[2][0] * stress[0] + /* J[2][1] * KJtD[1] + */ J[2][2] * stress[2];
        F[3] = /* J[3][0] * KJtD[0] + */ J[3][1] * stress[1] + J[3][2] * stress[2];
        F[4] = /* J[4][0] * KJtD[0] + J[4][1] * KJtD[1] + */ J[4][2] * stress[2];
        F[5] = /* J[5][0] * KJtD[0] + */ J[5][1] * stress[1] /* + J[5][2] * KJtD[2] */ ;

        // Since J has been "normalized" we need to multiply the force F by the area of the triangle to get the correct force
        F *= triangleInf[elementIndex].area;

        // store newly computed values for next time
        R_2_0.transpose(R_0_2);
        triangleInf[elementIndex].strainDisplacementMatrix = J;
        triangleInf[elementIndex].rotation = R_2_0;
        triangleInf[elementIndex].strain = strain;
        triangleInf[elementIndex].stress = stress;
        triangleInf[elementIndex].stiffness = K;
    }

    triangleInfo.endEdit();
}

/// Compute current stress
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStress(defaulttype::Vec<3,Real> &stress, Index elementIndex)
{
    Displacement D;
    StrainDisplacement J;
    defaulttype::Vec<3,Real> strain;
    Transformation R_0_2, R_2_0;
    const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    if (method == SMALL)
    {
        // classic linear elastic method
        R_0_2.identity();
        computeDisplacementSmall(D, elementIndex, p);
        if (_anisotropicMaterial)
            computeStrainDisplacement(J, elementIndex, Coord(0,0,0), (p[b]-p[a]), (p[c]-p[a]));
        else
            J = triangleInf[elementIndex].strainDisplacementMatrix;
        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[elementIndex].materialMatrix, strain);
    }
    else
    {
        // co-rotational method
        // first, compute rotation matrix into co-rotational frame
        computeRotationLarge( R_0_2, p, a, b, c);

        // then compute displacement in this frame
        computeDisplacementLarge(D, elementIndex, R_0_2, p);

        // and compute postions of a, b, c in the co-rotational frame
        Coord A = Coord(0, 0, 0);
        Coord B = R_0_2 * (p[b]-p[a]);
        Coord C = R_0_2 * (p[c]-p[a]);

        computeStrainDisplacement(J, elementIndex, A, B, C);
        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[elementIndex].materialMatrix, strain);
    }
    // store newly computed values for next time
    R_2_0.transpose(R_0_2);
    triangleInf[elementIndex].strainDisplacementMatrix = J;
    triangleInf[elementIndex].rotation = R_2_0;
    triangleInf[elementIndex].strain = strain;
    triangleInf[elementIndex].stress = stress;

    triangleInfo.endEdit();
}

// ----------------------------------------------------------------------------------------------------------------------------------------
// ---	Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
// ----------------------------------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &dir, const defaulttype::Vec<3,Real> &stress)
{
    defaulttype::Mat<3,3,Real> R, Rt;

    helper::vector<TriangleInformation>& triangleInf = *(this->triangleInfo.beginEdit());

    // transform 'dir' into local coordinates
    R = triangleInf[elementIndex].rotation;
    Rt.transpose(R);
    Coord dir_local = Rt * dir;
    dir_local[2] = 0; // project direction
    dir_local.normalize();

    // compute stress along specified direction 'dir'
    Real cos_theta = dir_local[0];
    Real sin_theta = dir_local[1];
    stress_along_dir = stress[0]*cos_theta*cos_theta + stress[1]*sin_theta*sin_theta + stress[2]*2*cos_theta*sin_theta;
    triangleInfo.endEdit();
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &dir, const defaulttype::Vec<3,Real> &stress)
{
    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord n = cross(x[b]-x[a],x[c]-x[a]);
    Coord dir_t = cross(dir,n);
    this->computeStressAlongDirection(stress_across_dir, elementIndex, dir_t, stress);
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &dir)
{
    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord n = cross(x[b]-x[a],x[c]-x[a]);
    Coord dir_t = cross(dir,n);
    this->computeStressAlongDirection(stress_across_dir, elementIndex, dir_t);
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &dir)
{
    defaulttype::Vec<3,Real> stress;
    this->computeStress(stress, elementIndex);
    this->computeStressAlongDirection(stress_along_dir, elementIndex, dir, stress);
}




// --------------------------------------------------------------------------------------
// --- Apply functions
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord &v, Real h, const VecCoord &x, const SReal &kFactor)
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering in applyStiffnessSmall." ;

    defaulttype::Mat<6,3,Real> J;
    defaulttype::Vec<3,Real> strain, stress;
    Displacement D, F;
    unsigned int nbTriangles=_topology->getNbTriangles();

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    for(unsigned int i=0; i<nbTriangles; i++)
    {
        Index a = _topology->getTriangle(i)[0];
        Index b = _topology->getTriangle(i)[1];
        Index c = _topology->getTriangle(i)[2];

        D[0] = x[a][0];
        D[1] = x[a][1];

        D[2] = x[b][0];
        D[3] = x[b][1];

        D[4] = x[c][0];
        D[5] = x[c][1];

        J = triangleInf[i].strainDisplacementMatrix;
        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[i].materialMatrix, strain);
        F = J * stress * triangleInf[i].area;

        v[a] += (Coord(-h*F[0], -h*F[1], 0)) * kFactor;
        v[b] += (Coord(-h*F[2], -h*F[3], 0)) * kFactor;
        v[c] += (Coord(-h*F[4], -h*F[5], 0)) * kFactor;
    }
    triangleInfo.endEdit();
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffness( VecCoord& v, Real h, const VecCoord& x, const SReal &kFactor )
{
    if (method == SMALL)
        applyStiffnessSmall( v, h, x, kFactor );
    else
        applyStiffnessLarge( v, h, x, kFactor );
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord &v, Real h, const VecCoord &x, const SReal &kFactor)
{

    if(DEBUG_TRIANGLEFEM_MSG)
        msg_info() << "Entering applyStiffnessLarge" ;

    defaulttype::Mat<6,3,Real> J;
    defaulttype::Vec<3,Real> strain, stress;
    MaterialStiffness K;
    Displacement D;
    Coord x_2;
    unsigned int nbTriangles = _topology->getNbTriangles();

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    for(unsigned int i=0; i<nbTriangles; i++)
    {
        Index a = _topology->getTriangle(i)[0];
        Index b = _topology->getTriangle(i)[1];
        Index c = _topology->getTriangle(i)[2];

        Transformation R_0_2;
        R_0_2.transpose(triangleInf[i].rotation);

        VecCoord disp;
        disp.resize(3);

        x_2 = R_0_2 * x[a];
        disp[0] = x_2;

        D[0] = x_2[0];
        D[1] = x_2[1];

        x_2 = R_0_2 * x[b];
        disp[1] = x_2;
        D[2] = x_2[0];
        D[3] = x_2[1];

        x_2 = R_0_2 * x[c];
        disp[2] = x_2;
        D[4] = x_2[0];
        D[5] = x_2[1];

        Displacement F;

        K = triangleInf[i].materialMatrix;
        J = triangleInf[i].strainDisplacementMatrix;

        computeStrain(strain, J, D);
        computeStress(stress, triangleInf[i].materialMatrix, strain);

        F[0] = J[0][0] * stress[0] + /* J[0][1] * KJtD[1] + */ J[0][2] * stress[2];
        F[1] = /* J[1][0] * KJtD[0] + */ J[1][1] * stress[1] + J[1][2] * stress[2];
        F[2] = J[2][0] * stress[0] + /* J[2][1] * KJtD[1] + */ J[2][2] * stress[2];
        F[3] = /* J[3][0] * KJtD[0] + */ J[3][1] * stress[1] + J[3][2] * stress[2];
        F[4] = /* J[4][0] * KJtD[0] + J[4][1] * KJtD[1] + */ J[4][2] * stress[2];
        F[5] = /* J[5][0] * KJtD[0] + */ J[5][1] * stress[1] /* + J[5][2] * KJtD[2] */ ;

        F *= triangleInf[i].area;

        v[a] += (triangleInf[i].rotation * Coord(-h*F[0], -h*F[1], 0)) * kFactor;
        v[b] += (triangleInf[i].rotation * Coord(-h*F[2], -h*F[3], 0)) * kFactor;
        v[c] += (triangleInf[i].rotation * Coord(-h*F[4], -h*F[5], 0)) * kFactor;
    }
    triangleInfo.endEdit();
}



// --------------------------------------------------------------------------------------
// --- Accumulate functions
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateDampingSmall(VecCoord&, Index )
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "TriangularFEMForceField::accumulateDampingSmall" ;
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceSmall( VecCoord &f, const VecCoord &p, Index elementIndex )
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangularFEMForceField::accumulateForceSmall"<<sendl;
#endif

    Displacement F;

    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    // compute force on element
    computeForce(F, elementIndex, p);

    f[a] += Coord( F[0], F[1], 0);
    f[b] += Coord( F[2], F[3], 0);
    f[c] += Coord( F[4], F[5], 0);
}


// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceLarge(VecCoord &f, const VecCoord &p, Index elementIndex )
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "TriangularFEMForceField::accumulateForceLarge" ;

    Displacement F;

    Index a = _topology->getTriangle(elementIndex)[0];
    Index b = _topology->getTriangle(elementIndex)[1];
    Index c = _topology->getTriangle(elementIndex)[2];

    // compute force on element (in the co-rotational space)
    computeForce( F, elementIndex, p);

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    // transform force back into global ref. frame
    f[a] += triangleInf[elementIndex].rotation * Coord(F[0], F[1], 0);
    f[b] += triangleInf[elementIndex].rotation * Coord(F[2], F[3], 0);
    f[c] += triangleInf[elementIndex].rotation * Coord(F[4], F[5], 0);

    triangleInfo.endEdit();
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateDampingLarge(VecCoord &, Index )
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "TriangularFEMForceField::accumulateDampingLarge" ;
}




// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    VecDeriv& f1 = *f.beginEdit();
    const VecCoord& x1 = x.getValue();

    int nbTriangles=_topology->getNbTriangles();

    f1.resize(x1.size());

    if(f_damping.getValue() != 0)
    {
        if(method == SMALL)
        {
            for( int i=0; i<nbTriangles; i+=3 )
            {
                accumulateForceSmall( f1, x1, i/3 );
                accumulateDampingSmall( f1, i/3 );
            }
        }
        else
        {
            for ( int i=0; i<nbTriangles; i+=3 )
            {
                accumulateForceLarge( f1, x1, i/3);
                accumulateDampingLarge( f1, i/3 );
            }
        }
    }
    else
    {
        if (method==SMALL)
        {
            for(int i=0; i<nbTriangles; i+=1)
            {
                accumulateForceSmall( f1, x1, i );
            }
        }
        else
        {
            for ( int i=0; i<nbTriangles; i+=1)
            {
                accumulateForceLarge( f1, x1, i);
            }
        }
    }
    f.endEdit();

    if (f_computePrincipalStress.getValue())
    {
        unsigned int nbTriangles=_topology->getNbTriangles();
        helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
        for(unsigned int i=0; i<nbTriangles; ++i)
            computePrincipalStress(i, triangleInf[i].stress);
        triangleInfo.endEdit();
    }

    //TODO(dmarchal 2017-05-03) I will remove this code soon !!!
    /*	if (f_fracturable.getValue())
    {
    // First Pass - Vertices Pass
    unsigned int nbPoints=_topology->getNbPoints();

    for( unsigned int i=0; i<nbPoints; i++ )
    {
    const sofa::helper::vector< unsigned int >& triangleNeighbors = _topology->getTrianglesAroundVertex(i);

    sofa::helper::vector< unsigned int >::const_iterator it = triangleNeighbors.begin();
    sofa::helper::vector< unsigned int >::const_iterator itEnd = triangleNeighbors.end();
    Coord meanStrainDirection, refStrainDirection;
    meanStrainDirection.clear();
    refStrainDirection.clear();

    bool b(true);

    while (it != itEnd)
    {
    if (b)
    {
    refStrainDirection = triangleInfo[*it].principalStrainDirection;
    if (refStrainDirection.norm() != 0.0)
    b=false;
    }

    (triangleInfo[*it].principalStrainDirection * refStrainDirection < 0)?
    meanStrainDirection -= triangleInfo[*it].principalStrainDirection : meanStrainDirection += triangleInfo[*it].principalStrainDirection;

    vertexInfo[i].sumEigenValues += triangleInfo[*it].eigenValue;

    ++it;
    }

    //meanStrainDirection.normalize();

    vertexInfo[i].meanStrainDirection = meanStrainDirection / (double)triangleNeighbors.size();
    vertexInfo[i].sumEigenValues = (double)vertexInfo[i].sumEigenValues / (double)triangleNeighbors.size();
    }


    // Second Pass - Edges Pass

    for(int i=0; i<_topology->getNbEdges(); i++ )
    edgeInfo[i].fracturable = false;

    if (nbPoints > 0)
    {
    double max = 0; //vertexInfo[0].sumEigenValues;
    int mostDeformableVertexIndex = -1; //0;

    for( unsigned int i=0; i<nbPoints; i++ )
    {
    bool vertexOnBorder = _topology->getTrianglesAroundVertex(i).size() < _topology->getEdgesAroundVertex(i).size() && _topology->getTrianglesAroundVertex(i).size() > 1;

    if (vertexOnBorder && vertexInfo[i].sumEigenValues > max)
    {
    mostDeformableVertexIndex = i;
    max = vertexInfo[i].sumEigenValues;
    }
    }

    //			std::vector< std::pair< double, unsigned int > > mostDeformableVertexIndexA(nbPoints);
    //			for( unsigned int i=0; i<nbPoints; i++ )
    //			{
    //				mostDeformableVertexIndexA[i].first = vertexInfo[i].sumEigenValues;
    //				mostDeformableVertexIndexA[i].second = i;
    //			}

    //			std::sort( mostDeformableVertexIndexA.begin(), mostDeformableVertexIndexA.end() );
    //		for( unsigned int i=0; i<nbPoints; i++ )
    //			{
    //				sout << mostDeformableVertexIndexA[i].first << " ";
    //			}
    //			sout << ""<<sendl;


    //			bool vertexOnBorder(false);
    //			int curIndex = nbPoints-1;

    //			while ((!vertexOnBorder)&&(curIndex>=0))
    //			{
    //				mostDeformableVertexIndex = mostDeformableVertexIndexA[curIndex].second;
    //				max=mostDeformableVertexIndexA[curIndex].first;
    // Check if the Vertex is on the border
    //				curIndex--;
    //			}

    // if (vertexOnBorder && (max > 0.05))

    const double THRESHOLD = 0.2;

    if ((mostDeformableVertexIndex!=-1) && (max > THRESHOLD))
    {
    //sout << "max=" << max << sendl;
    double minDotProduct = 1000.0;
    unsigned int fracturableIndex = 0;
    bool fracture(false);

    const sofa::helper::vector< unsigned int >& edgeNeighbors = _topology->getEdgesAroundVertex(mostDeformableVertexIndex);

    sofa::helper::vector< unsigned int >::const_iterator it = edgeNeighbors.begin();
    sofa::helper::vector< unsigned int >::const_iterator itEnd = edgeNeighbors.end();

    Index a;
    Index b;

    Coord n = vertexInfo[mostDeformableVertexIndex].meanStrainDirection;
    n.normalize();

    while (it != itEnd)
    {
    a = _topology->getEdge(*it)[0];
    b = _topology->getEdge(*it)[1];

    if (vertexInfo[mostDeformableVertexIndex].meanStrainDirection.norm() != 0.0)
    {
    Coord d = x[b]-x[a];
    d.normalize();
    if (fabs(n * d) < minDotProduct)
    {
    sofa::helper::vector< unsigned int > trianglesAroundEdge = _topology->getTrianglesAroundEdge(*it);
    if (trianglesAroundEdge.size() != 1)
    {

    //	bool bb(false);
    //	sofa::helper::vector< unsigned int >::iterator _it = trianglesAroundEdge.begin();
    //	sofa::helper::vector< unsigned int >::iterator _itEnd = trianglesAroundEdge.end();
    //	while (_it != _itEnd)
    //	{
    //		helper::fixed_array<unsigned int,3> edges = _topology->getEdgesInTriangle(*_it);

    //		int cptTest=0;
    //		for (int i=0; i<3; i++)
    //		{
    //			if (_topology->getTrianglesAroundEdge(edges[i]).size() < 2)
    //			{
    //				cptTest++;
    //			}
    //		}

    //		if (cptTest > 2)
    //		{
    //			if (max < 5 * THRESHOLD)
    //			{
    //				bb = true;
    //				break;
    //			}
    //		}

    //		++_it;
    //	}

    //	if (!bb)

    {
    minDotProduct = fabs(n * d);
    fracturableIndex = *it;
    fracture = true;
    }
    }
    }
    }

    ++it;
    }

    if (fracture) {
    //sout << "fracture at edge "<<fracturableIndex<<sendl;
    edgeInfo[fracturableIndex].fracturable = true;
    lastFracturedEdgeIndex = fracturableIndex;
    }
    }
    }
    }
    */

    //	sout << "EOF AddForce"<<sendl;
    /*	int nodeIdx=88;
    Transformation r,rt,I;
    getRotation(r,nodeIdx);
    rt.transpose(r);
    I=rt*r;

    serr << "rotation Matrix"<< r <<sendl;
    serr << "rotation Matrix"<< rt <<sendl;
    serr << "rotation Matrix"<< I <<sendl;*/

    /*	helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());
    int numNeiTri=_topology->getTrianglesAroundVertex(nodeIdx).size();
    for(int i=0;i<numNeiTri;i++)
    {
    int triIdx=_topology->getTrianglesAroundVertex(nodeIdx)[i];
    TriangleInformation *tinfo = &triangleInf[triIdx];
    Transformation r01,r21;
    r01=tinfo->initialTransformation;
    r21=tinfo->rotation*r01;
    serr << "rotation Matrix of"<< i << "triangle"<< r21 <<sendl;
    }*/
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    VecDeriv& df1 = *df.beginEdit();
    const VecDeriv& dx1 = dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    Real h=1;
    df1.resize(dx1.size());

    if (method == SMALL)
        applyStiffnessSmall( df1, h, dx1, kFactor );
    else
        applyStiffnessLarge( df1, h, dx1, kFactor );

    df.endEdit();
}




// --------------------------------------------------------------------------------------
// --- Display methods
// --------------------------------------------------------------------------------------
template<class DataTypes>
void TriangularFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    unsigned int nbTriangles=_topology->getNbTriangles();

    glDisable(GL_LIGHTING);
    if (!f_fracturable.getValue() && !this->showFracturableTriangles.getValue())
    {
        glBegin(GL_TRIANGLES);
        for(unsigned int i=0; i<nbTriangles; ++i)
        {
            Index a = _topology->getTriangle(i)[0];
            Index b = _topology->getTriangle(i)[1];
            Index c = _topology->getTriangle(i)[2];

            glColor4f(0,1,0,1);
            helper::gl::glVertexT(x[a]);
            glColor4f(0,0.5,0.5,1);
            helper::gl::glVertexT(x[b]);
            glColor4f(0,0,1,1);
            helper::gl::glVertexT(x[c]);
        }
        glEnd();
    }

    helper::vector<TriangleInformation>& triangleInf = *(triangleInfo.beginEdit());

    if (showStressVector.getValue() || showStressValue.getValue() || showFracturableTriangles.getValue())
    {
        for(unsigned int i=0; i<nbTriangles; ++i)
            computePrincipalStress(i, triangleInf[i].stress);
    }

    if (showStressVector.getValue())
    {
        glColor4f(1,0,1,1);
        glBegin(GL_LINES);
        for(unsigned int i=0; i<nbTriangles; ++i)
        {
            Index a = _topology->getTriangle(i)[0];
            Index b = _topology->getTriangle(i)[1];
            Index c = _topology->getTriangle(i)[2];
            Coord center = (x[a]+x[b]+x[c])/3;
            Coord d = triangleInf[i].principalStressDirection*2.5; //was 0.25
            helper::gl::glVertexT(center);
            helper::gl::glVertexT(center+d);
        }
        glEnd();
    }

    if (showStressValue.getValue())
    {
        helper::vector<VertexInformation>& vertexInf = *(vertexInfo.beginEdit());
        double minStress = numeric_limits<double>::max();
        double maxStress = 0.0;
        for ( unsigned int i = 0 ; i < vertexInf.size() ; i++)
        {
            core::topology::BaseMeshTopology::TrianglesAroundVertex triangles = _topology->getTrianglesAroundVertex(i);
            double averageStress = 0.0;
            double sumArea = 0.0;
            for ( unsigned int v = 0 ; v < triangles.size() ; v++)
            {
                if ( triangleInfo.getValue()[triangles[v]].area)
                {
                    averageStress+= ( fabs(triangleInfo.getValue()[triangles[v]].maxStress) * triangleInfo.getValue()[triangles[v]].area);
                    sumArea += triangleInfo.getValue()[triangles[v]].area;
                }
            }
            if (sumArea)
                averageStress /= sumArea;

            vertexInf[i].stress = averageStress;
            if (averageStress < minStress )
                minStress = averageStress;
            if (averageStress > maxStress)
                maxStress = averageStress;
        }

        helper::ColorMap::evaluator<double> evalColor = helper::ColorMap::getDefault()->getEvaluator(minStress, maxStress);
        glBegin(GL_TRIANGLES);
        for(unsigned int i=0; i<nbTriangles; ++i)
        {
            Index a = _topology->getTriangle(i)[0];
            Index b = _topology->getTriangle(i)[1];
            Index c = _topology->getTriangle(i)[2];

            glColor4fv(evalColor(vertexInf[a].stress).ptr());
            helper::gl::glVertexT(x[a]);
            glColor4fv(evalColor(vertexInf[b].stress).ptr());
            helper::gl::glVertexT(x[b]);
            glColor4fv(evalColor(vertexInf[c].stress).ptr());
            helper::gl::glVertexT(x[c]);
        }
        glEnd();
    }

    if (showFracturableTriangles.getValue())
    {
        Real maxDifference = numeric_limits<Real>::min();
        Real minDifference = numeric_limits<Real>::max();
        for (unsigned int i = 0 ; i < nbTriangles ; i++)
        {
            if (triangleInf[i].differenceToCriteria > 0)
            {
                if (triangleInf[i].differenceToCriteria > maxDifference)
                    maxDifference = triangleInf[i].differenceToCriteria;

                if (triangleInf[i].differenceToCriteria < minDifference)
                    minDifference = triangleInf[i].differenceToCriteria;
            }
        }

        glBegin(GL_TRIANGLES);
        for (unsigned int i = 0 ; i < nbTriangles ; i++)
        {
            if (triangleInf[i].differenceToCriteria > 0)
            {
                glColor4d( 0.4 + 0.4 * (triangleInf[i].differenceToCriteria - minDifference ) /  (maxDifference - minDifference) , 0.0 , 0.0, 0.5);

                Index a = _topology->getTriangle(i)[0];
                Index b = _topology->getTriangle(i)[1];
                Index c = _topology->getTriangle(i)[2];

                helper::gl::glVertexT(x[a]);
                helper::gl::glVertexT(x[b]);
                helper::gl::glVertexT(x[c]);
            }
        }
        glEnd();
    }
    triangleInfo.endEdit();
    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa


#endif //SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_INL
