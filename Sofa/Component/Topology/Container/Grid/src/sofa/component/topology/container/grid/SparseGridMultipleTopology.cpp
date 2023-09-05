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
#include <sofa/component/topology/container/grid/SparseGridMultipleTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa::component::topology::container::grid
{

int SparseGridMultipleTopologyClass = core::RegisterObject("Sparse grid in 3D")
        .addAlias("SparseGridMultiple")
        .add< SparseGridMultipleTopology >()
        ;


SparseGridMultipleTopology::SparseGridMultipleTopology( bool _isVirtual ) : SparseGridRamificationTopology(_isVirtual),
    _fileTopologies(initData(&_fileTopologies, type::vector< std::string >() , "fileTopologies", "All topology filenames")),
    _dataStiffnessCoefs(initData(&_dataStiffnessCoefs, type::vector< float >() , "stiffnessCoefs", "A stiffness coefficient for each topology filename")),
    _dataMassCoefs(initData(&_dataMassCoefs, type::vector< float >() , "massCoefs", "A mass coefficient for each topology filename")),
    _computeRamifications(initData(&_computeRamifications, true , "computeRamifications", "Are ramifications wanted?")),
    _erasePreviousCoef(initData(&_erasePreviousCoef, false , "erasePreviousCoef", "Does a new stiffness/mass coefficient replace the previous or blend half/half with it?"))
{
}


void SparseGridMultipleTopology::buildAsFinest()
{
    if (_fileTopologies.getValue().empty())
    {
        msg_error() << "No file topology provided";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if( _dataStiffnessCoefs.getValue().size() < _fileTopologies.getValue().size() )
    {
        msg_warning() << "SparseGridMultipleTopology: not enough stiffnessCoefs";
        for(unsigned i=_dataStiffnessCoefs.getValue().size(); i<_fileTopologies.getValue().size(); ++i)
            _dataStiffnessCoefs.beginEdit()->push_back( 1.0 );
        //           return;
    }

    if( _dataMassCoefs.getValue().size() < _fileTopologies.getValue().size() )
    {
        msg_warning() << "SparseGridMultipleTopology: not enough massCoefs\n";
        for(unsigned i=_dataMassCoefs.getValue().size(); i<_fileTopologies.getValue().size(); ++i)
            _dataMassCoefs.beginEdit()->push_back( 1.0 );
        // 			return;
    }

    const unsigned regularGridsSize = _regularGrids.size();

    if (regularGridsSize < _fileTopologies.getValue().size())
    {
        for (unsigned int i = 0; i < _fileTopologies.getValue().size() - regularGridsSize; ++i)
        {
            _regularGrids.push_back(sofa::core::objectmodel::New< RegularGridTopology >());
        }
    }
    else
    {
        for (unsigned int i = 0; i < regularGridsSize - _fileTopologies.getValue().size(); ++i)
        {
            _regularGrids[i + _regularGrids.size()].reset();
        }

        _regularGrids.resize(_fileTopologies.getValue().size());
    }

    _regularGridTypes.resize(_fileTopologies.getValue().size());

    type::vector< helper::io::Mesh*> meshes(_fileTopologies.getValue().size());


    SReal xMing = std::numeric_limits<SReal>::max();
    SReal xMaxg = std::numeric_limits<SReal>::lowest();
    SReal yMing = std::numeric_limits<SReal>::max();
    SReal yMaxg = std::numeric_limits<SReal>::lowest();
    SReal zMing = std::numeric_limits<SReal>::max();
    SReal zMaxg = std::numeric_limits<SReal>::lowest();

    for(unsigned i=0; i<_fileTopologies.getValue().size(); ++i)
    {

        std::string filename = _fileTopologies.getValue()[i];

        msg_info() << "SparseGridMultipleTopology open " << filename;

        if (! sofa::helper::system::DataRepository.findFile ( filename ))
            continue;

        if( (filename.length() > 4 && filename.compare(filename.length()-4, 4, ".obj")==0) || (filename.length() > 6 && filename.compare(filename.length()-6, 6, ".trian")==0))
        {
            meshes[i] = helper::io::Mesh::Create(filename.c_str());

            if(meshes[i] == nullptr)
            {
                msg_error() << "SparseGridTopology: loading mesh " << filename << " failed.";
                return;
            }


            SReal xMin, xMax, yMin, yMax, zMin, zMax;
            computeBoundingBox(meshes[i]->getVertices(), xMin, xMax, yMin, yMax, zMin, zMax);


            if( xMin < xMing ) xMing=xMin;
            if( yMin < yMing ) yMing=yMin;
            if( zMin < zMing ) zMing=zMin;
            if( xMax > xMaxg ) xMaxg=xMax;
            if( yMax > yMaxg ) yMaxg=yMax;
            if( zMax > zMaxg ) zMaxg=zMax;
        }
    }

    if( _min.getValue()== type::Vec3() && _max.getValue()== type::Vec3())
    {
        // increase the box a little
        type::Vec3 diff ( xMaxg-xMing, yMaxg - yMing, zMaxg - zMing );
        diff /= 100.0;

        _min.setValue(type::Vec3( xMing - diff[0], yMing - diff[1], zMing - diff[2] ));
        _max.setValue(type::Vec3( xMaxg + diff[0], yMaxg + diff[1], zMaxg + diff[2] ));
    }

    for(unsigned i=0; i<_fileTopologies.getValue().size(); ++i)
    {
        if(meshes[i]) buildFromTriangleMesh(meshes[i],i);
    }


    type::vector<Type> regularGridTypes; // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    type::vector< float > regularStiffnessCoefs,regularMassCoefs;

    assembleRegularGrids( regularGridTypes, regularStiffnessCoefs, regularMassCoefs );

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    _stiffnessCoefs.resize( this->getNbHexahedra());
    _massCoefs.resize( this->getNbHexahedra());

    for(size_t i=0; i<this->getNbHexahedra(); ++i)
    {
        _stiffnessCoefs[i] = regularStiffnessCoefs[ this->_indicesOfCubeinRegularGrid[i] ];
        _massCoefs[i] = regularMassCoefs[ this->_indicesOfCubeinRegularGrid[i] ];
    }

    if(_computeRamifications.getValue())
    {

        if( _finestConnectivity.getValue() || this->isVirtual || _nbVirtualFinerLevels.getValue() > 0 )
        {
            // find the connexion graph between the finest hexahedra
            findConnexionsAtFinestLevel();
        }

        if( _finestConnectivity.getValue() )
        {

            buildRamifiedFinestLevel();
        }
    }
}


void SparseGridMultipleTopology::buildFromTriangleMesh(helper::io::Mesh* mesh, unsigned fileIdx)
{
    _regularGrids[fileIdx]->setSize(this->getNx(),this->getNy(),this->getNz());
    _regularGrids[fileIdx]->setPos(this->getXmin(),this->getXmax(),this->getYmin(),this->getYmax(),this->getZmin(),this->getZmax());

    voxelizeTriangleMesh(mesh, _regularGrids[fileIdx], _regularGridTypes[fileIdx]);

    delete mesh;
}


void SparseGridMultipleTopology::assembleRegularGrids(type::vector<Type>& regularGridTypes,type::vector< float >& regularStiffnessCoefs,type::vector< float >& regularMassCoefs)
{
    _regularGrid->setSize(getNx(),getNy(),getNz());
    _regularGrid->setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    regularGridTypes.resize( _regularGridTypes[0].size() );
    regularGridTypes.fill(OUTSIDE);
    regularStiffnessCoefs.resize( _regularGridTypes[0].size() );
    regularMassCoefs.resize( _regularGridTypes[0].size() );

    for(size_t i=0; i<_regularGrids.size(); ++i)
    {
        for(size_t w=0; w<_regularGrids[i]->getNbHexahedra(); ++w)
        {
            if( _regularGridTypes[i][w] == INSIDE || (_regularGridTypes[i][w] == BOUNDARY && !this->_fillWeighted.getValue()) )
            {
                regularGridTypes[w] = INSIDE;
                regularStiffnessCoefs[w] = _dataStiffnessCoefs.getValue()[i];
                regularMassCoefs[w] = _dataMassCoefs.getValue()[i];
            }
            else if(  _regularGridTypes[i][w] == BOUNDARY && this->_fillWeighted.getValue() )
            {
                if( regularGridTypes[w] != INSIDE ) regularGridTypes[w] = BOUNDARY;

                regularStiffnessCoefs[w] = (float)(_erasePreviousCoef.getValue()?_dataStiffnessCoefs.getValue()[i]:(regularStiffnessCoefs[w]+_dataStiffnessCoefs.getValue()[i]) * .5f);
                regularMassCoefs[w] = (float)(_erasePreviousCoef.getValue()?_dataMassCoefs.getValue()[i]:(regularMassCoefs[w]+_dataMassCoefs.getValue()[i]) * .5f);
            }
        }
    }
}


void SparseGridMultipleTopology::buildVirtualFinerLevels()
{
    int nb = _nbVirtualFinerLevels.getValue();

    _virtualFinerLevels.resize(nb);

    int newnx=n.getValue()[0],newny=n.getValue()[1],newnz=n.getValue()[2];
    for( int i=0; i<nb; ++i)
    {
        newnx = (newnx-1)*2+1;
        newny = (newny-1)*2+1;
        newnz = (newnz-1)*2+1;
    }

    SparseGridMultipleTopology::SPtr sgmt = sofa::core::objectmodel::New< SparseGridMultipleTopology >(true);

    _virtualFinerLevels[0] = sgmt;
    _virtualFinerLevels[0]->setNx( newnx );
    _virtualFinerLevels[0]->setNy( newny );
    _virtualFinerLevels[0]->setNz( newnz );
    _virtualFinerLevels[0]->setMin( _min.getValue() );
    _virtualFinerLevels[0]->setMax( _max.getValue() );
    _virtualFinerLevels[0]->_fillWeighted.setValue( _fillWeighted.getValue() );
    std::stringstream nameg; nameg << "virtual grid "<< 0;
    _virtualFinerLevels[0]->setName( nameg.str().c_str() );
    this->addSlave(_virtualFinerLevels[0]); //->setContext( this->getContext() );
    sgmt->_erasePreviousCoef.setValue(_erasePreviousCoef.getValue());
    _virtualFinerLevels[0]->load(this->fileTopology.getValue().c_str());
    sgmt->_fileTopologies.setValue(this->_fileTopologies.getValue());
    sgmt->_dataStiffnessCoefs.setValue(this->_dataStiffnessCoefs.getValue());
    sgmt->_dataMassCoefs.setValue(this->_dataMassCoefs.getValue());
    sgmt->_finestConnectivity.setValue( _finestConnectivity.getValue() );
    _virtualFinerLevels[0]->init();

    std::stringstream tmpStr;
    tmpStr<<"SparseGridTopology "<<getName()<<" buildVirtualFinerLevels : ";
    tmpStr<<"("<<newnx<<"x"<<newny<<"x"<<newnz<<") -> "<< _virtualFinerLevels[0]->getNbHexahedra() <<" elements , ";

    for(int i=1; i<nb; ++i)
    {
        _virtualFinerLevels[i] = sofa::core::objectmodel::New< SparseGridMultipleTopology >(true);

        std::stringstream nameg2; nameg2 << "virtual grid "<< i;
        this->addSlave(_virtualFinerLevels[i]);
        _virtualFinerLevels[i]->setName( nameg2.str().c_str() );
        _virtualFinerLevels[i]->setFinerSparseGrid(_virtualFinerLevels[i-1].get());
        _virtualFinerLevels[i]->init();

        tmpStr<<"("<<_virtualFinerLevels[i]->getNx()<<"x"<<_virtualFinerLevels[i]->getNy()<<"x"<<_virtualFinerLevels[i]->getNz()<<") -> "<< _virtualFinerLevels[i]->getNbHexahedra() <<" elements , ";
    }

    msg_info() << tmpStr.str();
    this->setFinerSparseGrid(_virtualFinerLevels[nb-1].get());
}

} // namespace sofa::component::topology::container::grid

