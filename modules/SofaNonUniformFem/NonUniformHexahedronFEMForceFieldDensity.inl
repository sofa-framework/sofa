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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_INL
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaNonUniformFem/NonUniformHexahedronFEMForceFieldDensity.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/FileRepository.h>

#include <fstream>


namespace sofa
{

namespace component
{

namespace forcefield
{

using std::set;
using namespace sofa::defaulttype;



template <class DataTypes>
void NonUniformHexahedronFEMForceFieldDensity<DataTypes>::init()
{
    //  	serr<<"NonUniformHexahedronFEMForceFieldDensity<DataTypes>::init()"<<sendl;

    if(this->_alreadyInit)return;
    else this->_alreadyInit=true;

    // 	NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init();

    this->core::behavior::ForceField<DataTypes>::init();

    if( this->getContext()->getMeshTopology()==NULL )
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a Topology."<<sendl;
        return;
    }

    this->_mesh = dynamic_cast<sofa::core::topology::BaseMeshTopology*>(this->getContext()->getMeshTopology());
    if ( this->_mesh==NULL)
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a MeshTopology."<<sendl;
        return;
    }
#ifdef SOFA_NEW_HEXA
    else if( this->_mesh->getNbHexahedra()<=0 )
#else
    else if( this->_mesh->getNbCubes()<=0 )
#endif // SOFA_NEW_HEXA
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a hexahedric MeshTopology."<<sendl;
        serr << this->_mesh->getName()<<sendl;
        serr << this->_mesh->getTypeName()<<sendl;
        serr<<this->_mesh->getNbPoints()<<sendl;
        return;
    }

    this->_sparseGrid = dynamic_cast<topology::SparseGridTopology*>(this->_mesh);



    if (this->_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        this->_initialPoints.setValue(p);
    }

    this->_materialsStiffnesses.resize(this->getIndexedElements()->size() );
    this->_rotations.resize( this->getIndexedElements()->size() );
    this->_rotatedInitialElements.resize(this->getIndexedElements()->size());
    // 	stiffnessFactor.resize(this->getIndexedElements()->size());


    // 	NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init();

    // verify if it is wanted and possible to compute non-uniform stiffness
    if( !this->_nbVirtualFinerLevels.getValue() || !this->_sparseGrid || this->_sparseGrid->getNbVirtualFinerLevels() < this->_nbVirtualFinerLevels.getValue()  )
    {
        // 		this->_nbVirtualFinerLevels.setValue(0);
        serr<<"Conflict in nb of virtual levels between ForceField "<<this->getName()<<" and SparseGrid "<<this->_sparseGrid->getName()<<" -> classical uniform properties are used" << sendl;
    }
    else
    {

    }



    this->_elementStiffnesses.beginEdit()->resize(this->getIndexedElements()->size());
    this->_elementMasses.beginEdit()->resize(this->getIndexedElements()->size());

    //Load Gray scale density from RAW file

    std::string path = densityFile.getFullPath();

    if (!densityFile.getValue().empty() && sofa::helper::system::DataRepository.findFile(path))
    {
        densityFile.setValue(sofa::helper::system::DataRepository.getFile(densityFile.getValue()));
        FILE *file = fopen( densityFile.getValue().c_str(), "r" );
        voxels.resize(dimensionDensityFile.getValue()[2]);
        for (unsigned int z=0; z<dimensionDensityFile.getValue()[2]; ++z)
        {
            voxels[z].resize(dimensionDensityFile.getValue()[0]);
            for (unsigned int x=0; x<dimensionDensityFile.getValue()[0]; ++x)
            {
                voxels[z][x].resize(dimensionDensityFile.getValue()[1]);
                for (unsigned int y=0; y<dimensionDensityFile.getValue()[1]; ++y)
                {
                    voxels[z][x][y] = getc(file);
                }
            }
        }

        fclose(file);
    }
    //////////////////////


    if (this->f_method.getValue() == "large")
        this->setMethod(HexahedronFEMForceFieldT::LARGE);
    else if (this->f_method.getValue() == "polar")
        this->setMethod(HexahedronFEMForceFieldT::POLAR);


    for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
    {


        Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            nodes[w] = this->_initialPoints.getValue()[(*this->getIndexedElements())[i][this->_indices[w]]];
#else
            nodes[w] = this->_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];
#endif


        typename HexahedronFEMForceFieldT::Transformation R_0_1;

        if( this->method == HexahedronFEMForceFieldT::LARGE )
        {
            Coord horizontal;
            horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
            Coord vertical;
            vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
            computeRotationLarge( R_0_1, horizontal,vertical);
        }
        else
            computeRotationPolar( R_0_1, nodes);

        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            this->_rotatedInitialElements[i][w] = R_0_1*this->_initialPoints.getValue()[(*this->getIndexedElements())[i][this->_indices[w]]];
#else
            this->_rotatedInitialElements[i][w] = R_0_1*this->_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];
#endif

        computeCoarseElementStiffness( (*this->_elementStiffnesses.beginEdit())[i],
                (*this->_elementMasses.beginEdit())[i],i,0);
    }
    //////////////////////


    // 	post-traitement of non-uniform stiffness
    if( this->_nbVirtualFinerLevels.getValue() )
    {
        this->_sparseGrid->setNbVirtualFinerLevels(0);
        //delete undesirable sparsegrids and hexa
        for(int i=0; i<this->_sparseGrid->getNbVirtualFinerLevels(); ++i)
            delete this->_sparseGrid->_virtualFinerLevels[i];
        this->_sparseGrid->_virtualFinerLevels.resize(0);
    }



    if(this->_useMass.getValue() )
    {

        MassT::init();
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );


        int i=0;
        for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            Vec<8,Coord> nodes;
            for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
                nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];
#else
                nodes[w] = this->_initialPoints.getValue()[(*it)[w]];
#endif
            // volume of a element
            Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

            volume *= (Real) (this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5:1.0);

            // mass of a particle...
            Real mass = Real (( volume * this->_density.getValue() ) / 8.0 );

            // ... is added to each particle of the element
            for(int w=0; w<8; ++w)
                this->_particleMasses[ (*it)[w] ] += mass;
        }
    }
    else
    {
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );

        Real mass = this->_totalMass.getValue() / Real(this->getIndexedElements()->size());
        for(unsigned i=0; i<this->_particleMasses.size(); ++i)
            this->_particleMasses[ i ] = mass;
    }

}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


template<class DataTypes>
void NonUniformHexahedronFEMForceFieldDensity<DataTypes>::computeCoarseElementStiffness( ElementStiffness &coarseElement, ElementMass &coarseMassElement, const int elementIndice,  int level)
{


    if (level == this->_nbVirtualFinerLevels.getValue())
    {

        //Get the 8 indices of the coarser Hexa
        const helper::fixed_array<unsigned int,8>& points = this->_sparseGrid->_virtualFinerLevels[0]->getHexahedra()[elementIndice];
        //Get the 8 points of the coarser Hexa
        helper::fixed_array<Coord,8> nodes;
#ifndef SOFA_NEW_HEXA
        for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[0]->getPointPos(points[this->_indices[k]]);
#else
        for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[0]->getPointPos(points[k]);
#endif


        //       //given an elementIndice, find the 8 others from the sparse grid
        //       //compute MaterialStiffness
        MaterialStiffness mat;

        double grayScale=1.0;
        if (!densityFile.getValue().empty())
        {
            int indexInRegularGrid = this->_sparseGrid->_virtualFinerLevels[0]->_indicesOfCubeinRegularGrid[elementIndice];
            const Vector3 coordinates = this->_sparseGrid->_virtualFinerLevels[0]->_regularGrid.getCubeCoordinate(indexInRegularGrid);
            const Vector3 factor = Vector3(
                    dimensionDensityFile.getValue()[0]/((SReal)this->_sparseGrid->_virtualFinerLevels[0]->_regularGrid.getNx()),
                    dimensionDensityFile.getValue()[1]/((SReal)this->_sparseGrid->_virtualFinerLevels[0]->_regularGrid.getNy()),
                    dimensionDensityFile.getValue()[2]/((SReal)this->_sparseGrid->_virtualFinerLevels[0]->_regularGrid.getNz())
                    );
            if (this->_sparseGrid->_virtualFinerLevels[0]->getVoxel((unsigned int)(factor[0]*coordinates[0]), (unsigned int)(factor[1]*coordinates[1]), (unsigned int)(factor[2]*coordinates[2])))
            {
                grayScale = 1+10*exp(1-256/((float)(voxels[(int)(factor[2]*coordinates[2])][(int)(factor[0]*coordinates[0])][(int)(factor[1]*coordinates[1])])));
            }
            //       sout << grayScale << " "<<sendl;
        }
        computeMaterialStiffness(mat,  this->f_youngModulus.getValue()*grayScale,this->f_poissonRatio.getValue());

        //Nodes are found using Sparse Grid
        HexahedronFEMForceFieldAndMassT::computeElementStiffness(coarseElement,mat,nodes,elementIndice, this->_sparseGrid->_virtualFinerLevels[0]->getType(elementIndice)==topology::SparseGridTopology::BOUNDARY?.5:1.0); // classical stiffness
        HexahedronFEMForceFieldAndMassT::computeElementMass(coarseMassElement,nodes,elementIndice,this->_sparseGrid->_virtualFinerLevels[0]->getType(elementIndice)==topology::SparseGridTopology::BOUNDARY?.5:1.0);

    }
    else
    {
        helper::fixed_array<int,8> finerChildren;
        if (level == 0)
        {
            finerChildren = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
        }
        else
        {
            finerChildren = this->_sparseGrid->_virtualFinerLevels[this->_nbVirtualFinerLevels.getValue()-level]->_hierarchicalCubeMap[elementIndice];
        }

        //     serr<<finerChildren<<""<<sendl;
        //Get the 8 points of the coarser Hexa
        for ( int i=0; i<8; ++i)
        {
            if (finerChildren[i] != -1)
            {
                ElementStiffness childElement;
                ElementMass childMassElement;
                computeCoarseElementStiffness(childElement, childMassElement, finerChildren[i], level+1);
                this->addFineToCoarse(coarseElement, childElement, i);
                this->addFineToCoarse(coarseMassElement, childMassElement, i);
            }
        }
    }
}


template<class DataTypes>
void NonUniformHexahedronFEMForceFieldDensity<DataTypes>::computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio)
{
    m[0][0] = m[1][1] = m[2][2] = 1;
    m[0][1] = m[0][2] = m[1][0]= m[1][2] = m[2][0] =  m[2][1] = (Real)(poissonRatio/(1-poissonRatio));
    m[0][3] = m[0][4] =	m[0][5] = 0;
    m[1][3] = m[1][4] =	m[1][5] = 0;
    m[2][3] = m[2][4] =	m[2][5] = 0;
    m[3][0] = m[3][1] = m[3][2] = m[3][4] =	m[3][5] = 0;
    m[4][0] = m[4][1] = m[4][2] = m[4][3] =	m[4][5] = 0;
    m[5][0] = m[5][1] = m[5][2] = m[5][3] =	m[5][4] = 0;
    m[3][3] = m[4][4] = m[5][5] = (Real)((1-2*poissonRatio)/(2*(1-poissonRatio)));
    m *= (Real)((youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio)));
    // S = [ U V V 0 0 0 ]
    //     [ V U V 0 0 0 ]
    //     [ V V U 0 0 0 ]
    //     [ 0 0 0 W 0 0 ]
    //     [ 0 0 0 0 W 0 ]
    //     [ 0 0 0 0 0 W ]
    // with U = y * (1-p)/( (1+p)(1-2p))
    //      V = y *    p /( (1+p)(1-2p))
    //      W = y *  1   /(2(1+p)) = (U-V)/2
}

template<class DataTypes>
void NonUniformHexahedronFEMForceFieldDensity<DataTypes>::drawSphere(double r, int lats, int longs, const Coord &pos)
{
    int i, j;
    for(i = 0; i <= lats; i++)
    {
        double lat0 = M_PI * (-0.5 + (double) (i - 1) / lats);
        double z0  = sin(lat0);
        double zr0 =  cos(lat0);

        double lat1 = M_PI * (-0.5 + (double) i / lats);
        double z1 = sin(lat1);
        double zr1 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for(j = 0; j <= longs; j++)
        {
            double lng = 2 * M_PI * (double) (j - 1) / longs;
            double x = cos(lng);
            double y = sin(lng);

            glNormal3f((float)(pos[0] + r*x * zr0), (float)(pos[1] + r*y * zr0), (float)(pos[2] + r*z0));
            glVertex3f((float)(pos[0] + r*x * zr0), (float)(pos[1] + r*y * zr0), (float)(pos[2] + r*z0));
            glNormal3f((float)(pos[0] + r*x * zr1), (float)(pos[1] + r*y * zr1), (float)(pos[2] + r*z1));
            glVertex3f((float)(pos[0] + r*x * zr1), (float)(pos[1] + r*y * zr1), (float)(pos[2] + r*z1));
        }
        glEnd();
    }
}


template<class DataTypes>
void NonUniformHexahedronFEMForceFieldDensity<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (this->getIndexedElements()->size() == 0) return;


    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_LIGHTING);

    std::map< unsigned int , std::pair<unsigned int, double> > stiffnessDraw;

    typename VecElement::const_iterator it;
    int hexa_elem;

    double radius=std::min(
            (x[ (*this->getIndexedElements())[0][1] ][0] - x[ (*this->getIndexedElements())[0][0] ][0]) ,
            std::min( (x[ (*this->getIndexedElements())[0][3] ][1] - x[ (*this->getIndexedElements())[0][0] ][1]) ,
                    (x[ (*this->getIndexedElements())[0][4] ][2] - x[ (*this->getIndexedElements())[0][0] ][2]) )
            )/8.0;

    for(it = this->getIndexedElements()->begin(), hexa_elem = 0 ; it != this->getIndexedElements()->end() ; ++it, ++hexa_elem)
    {
        for (unsigned int elem=0; elem<8; ++elem)
        {
            double s=0;
            const unsigned index = (*it)[elem];
            for (unsigned int i=0; i<24; ++i)
            {
                s+= fabs(this->_elementStiffnesses.getValue()[hexa_elem][i][elem+0]) +
                    fabs(this->_elementStiffnesses.getValue()[hexa_elem][i][elem+1]) +
                    fabs(this->_elementStiffnesses.getValue()[hexa_elem][i][elem+2]);
            }
            if (stiffnessDraw.find( index ) != stiffnessDraw.end())
                stiffnessDraw[index]=std::make_pair(stiffnessDraw[index].first+1,s+stiffnessDraw[index].second);
            else
                stiffnessDraw[index]=std::make_pair(1,s);
        }
    }

    std::map< unsigned int ,  std::pair<unsigned int, double> >::iterator it_stiff;
    double max=-1;
    for (it_stiff = stiffnessDraw.begin(); it_stiff != stiffnessDraw.end(); ++it_stiff)
    {
        (*it_stiff).second.second /= (*it_stiff).second.first;
        if (max < 0 || max < (*it_stiff).second.second ) max = (*it_stiff).second.second;
    }

    for (it_stiff = stiffnessDraw.begin(); it_stiff != stiffnessDraw.end(); ++it_stiff)
    {
        glColor4f((float)((*it_stiff).second.second/max), 0.0f, (float)(1.0f-(*it_stiff).second.second/max),1.0f);
        drawSphere(radius*(1+2*(*it_stiff).second.second/max),10,10,x[ (*it_stiff).first ]);
    }
    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    //   if(this->_sparseGrid )
    //     glDisable(GL_BLEND);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_INL
