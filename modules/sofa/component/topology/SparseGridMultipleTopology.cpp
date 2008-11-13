
#include <sofa/component/topology/SparseGridMultipleTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace component
{

namespace topology
{

using std::cerr; using std::cout; using std::endl;

SOFA_DECL_CLASS(SparseGridMultipleTopology)

int SparseGridMultipleTopologyClass = core::RegisterObject("Sparse grid in 3D")
        .addAlias("SparseGridMultiple")
        .add< SparseGridMultipleTopology >()
        ;


void SparseGridMultipleTopology::buildAsFinest(  )
{
    cerr<<"SparseGridMultipleTopology::buildAsFinest\n";

    if( _dataStiffnessCoefs.getValue().size() < _fileTopologies.getValue().size() )
    {
        cerr<<"SparseGridMultipleTopology  ERROR: not enough stiffnessCoefs\n";
        return;
    }



    _regularGrids.resize(  _fileTopologies.getValue().size() );
    _regularGridTypes.resize(  _fileTopologies.getValue().size() );



    helper::vector< helper::io::Mesh*> meshes(_fileTopologies.getValue().size());


    SReal xMing=99999999, xMaxg=-99999999, yMing=99999999, yMaxg=-99999999, zMing=99999999, zMaxg=-99999999;


    for(unsigned i=0; i<_fileTopologies.getValue().size(); ++i)
    {

        std::string filename = _fileTopologies.getValue()[i];

        cerr<<"SparseGridMultipleTopology open "<<filename<<endl;

        if (! sofa::helper::system::DataRepository.findFile ( filename ))
            continue;

        if( (filename.length() > 4 && filename.compare(filename.length()-4, 4, ".obj")==0) || (filename.length() > 6 && filename.compare(filename.length()-6, 6, ".trian")==0))
        {
            meshes[i] = helper::io::Mesh::Create(filename.c_str());

            if(meshes[i] == NULL)
            {
                std::cerr << "SparseGridTopology: loading mesh " << filename << " failed." <<std::endl;
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



    // increase the box a little
    Vector3 diff ( xMaxg-xMing, yMaxg - yMing, zMaxg - zMing );
    diff /= 100.0;

    _min.setValue(Vector3( xMing - diff[0], yMing - diff[1], zMing - diff[2] ));
    _max.setValue(Vector3( xMaxg + diff[0], yMaxg + diff[1], zMaxg + diff[2] ));


    for(unsigned i=0; i<_fileTopologies.getValue().size(); ++i)
    {
        if(meshes[i]) buildFromTriangleMesh(meshes[i],i);
    }


    helper::vector<Type> regularGridTypes; // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
    helper::vector< float > regularStiffnessCoefs;

    assembleRegularGrids( regularGridTypes, regularStiffnessCoefs );

    buildFromRegularGridTypes(_regularGrid, regularGridTypes);

    _stiffnessCoefs.resize( this->getNbHexas());
    for(int i=0; i<this->getNbHexas(); ++i)
    {
        _stiffnessCoefs[i] = regularStiffnessCoefs[ this->_indicesOfCubeinRegularGrid[i] ];
    }



    findConnexionsAtFinestLevel();


}


void SparseGridMultipleTopology::buildFromTriangleMesh(helper::io::Mesh* mesh, unsigned fileIdx)
{
    _regularGrids[fileIdx].setSize(getNx(),getNy(),getNz());
    _regularGrids[fileIdx].setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    voxelizeTriangleMesh(mesh, _regularGrids[fileIdx], _regularGridTypes[fileIdx]);

    delete mesh;
}



void SparseGridMultipleTopology::assembleRegularGrids(helper::vector<Type>& regularGridTypes,helper::vector< float >& regularStiffnessCoefs)
{
    _regularGrid.setSize(getNx(),getNy(),getNz());
    _regularGrid.setPos(getXmin(),getXmax(),getYmin(),getYmax(),getZmin(),getZmax());

    regularGridTypes.resize( _regularGridTypes[0].size() );
    regularGridTypes.fill(OUTSIDE);
    regularStiffnessCoefs.resize( _regularGridTypes[0].size() );

    for(unsigned i=0; i<_regularGrids.size(); ++i)
    {
        for(int w=0; w<_regularGrids[i].getNbHexas(); ++w)
        {
            if( _regularGridTypes[i][w] == INSIDE )
            {
                regularGridTypes[w] = INSIDE;
                regularStiffnessCoefs[w] = _dataStiffnessCoefs.getValue()[i];
            }
            else if(  _regularGridTypes[i][w] == BOUNDARY && regularGridTypes[w] != INSIDE )
            {
                regularGridTypes[w] = BOUNDARY;
                regularStiffnessCoefs[w] = _dataStiffnessCoefs.getValue()[i] /** .5*/;
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

    SparseGridMultipleTopology* sgmt = new SparseGridMultipleTopology(true);;

    _virtualFinerLevels[0] = sgmt;
    _virtualFinerLevels[0]->setNx( newnx );
    _virtualFinerLevels[0]->setNy( newny );
    _virtualFinerLevels[0]->setNz( newnz );
// 		  _virtualFinerLevels[0]->load(this->fileTopology.getValue().c_str());
    sgmt->_fileTopologies.setValue(this->_fileTopologies.getValue());
    sgmt->_dataStiffnessCoefs.setValue(this->_dataStiffnessCoefs.getValue());
    _virtualFinerLevels[0]->init();

    cerr<<"SparseGridTopology "<<getName()<<" buildVirtualFinerLevels : ";
    cerr<<"("<<newnx<<"x"<<newny<<"x"<<newnz<<") -> "<< _virtualFinerLevels[0]->getNbHexas() <<" elements , ";

    for(int i=1; i<nb; ++i)
    {
        _virtualFinerLevels[i] = new SparseGridMultipleTopology(true);


        _virtualFinerLevels[i]->setFinerSparseGrid(_virtualFinerLevels[i-1]);

        _virtualFinerLevels[i]->init();


        cerr<<"("<<_virtualFinerLevels[i]->getNx()<<"x"<<_virtualFinerLevels[i]->getNy()<<"x"<<_virtualFinerLevels[i]->getNz()<<") -> "<< _virtualFinerLevels[i]->getNbHexas() <<" elements , ";
    }

    cerr<<endl;

    this->setFinerSparseGrid(_virtualFinerLevels[nb-1]);


}



}
}
}
