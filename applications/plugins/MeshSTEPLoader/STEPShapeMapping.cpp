#include "STEPShapeMapping.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{
using namespace sofa::component::topology;
using namespace sofa::component::topology::container::constant;

using namespace sofa::component::engine;
using namespace sofa::component::loader;

int STEPShapeExtractorClass = core::RegisterObject("Extract a shape from a MeshSTEPLoader according to a shape number.")
        .add< STEPShapeExtractor>(true);


STEPShapeExtractor::STEPShapeExtractor(MeshSTEPLoader* loader, MeshTopology* topology):
    shapeNumber(initData(&shapeNumber,"shapeNumber", "Shape number to be loaded" ) )
    ,indexBegin(initData(&indexBegin,(unsigned int)0,"indexBegin","The begin index for this shape with respect to the global mesh",true,true))
    ,indexEnd(initData(&indexEnd,(unsigned int)0,"indexEnd","The end index for this shape with respect to the global mesh",true,true))
    ,loader(initLink("input","Input MeshSTEPLoader to map"), loader)
    ,topology(initLink("output", "Output MeshTopology to map"), topology)
{
}


void STEPShapeExtractor::init()
{
    MeshSTEPLoader* input = loader.get();
    MeshTopology* output = topology.get();



    if( input == NULL || output == NULL )
    {
        msg_error() << "init failed! NULL pointers.";
        return;
    }

    addInput(&input->d_positions);
    addInput(&input->d_triangles);
    addInput(&input->_uv);

    addOutput(&output->seqPoints);
    addOutput(&output->seqTriangles);
    addOutput(&output->seqUVs);
}


void STEPShapeExtractor::doUpdate()
{
    MeshSTEPLoader* input = loader.get();
    MeshTopology* output = topology.get();

    if( input == NULL || output == NULL )
    {
        msg_error() << "init failed! NULL pointers.";
        return;
    }

    const type::vector<sofa::type::Vec3>& positionsI = input->d_positions.getValue();
    const type::vector<Triangle >& trianglesI = input->d_triangles.getValue();
    const type::vector<sofa::type::Vec2>& uvI = input->_uv.getValue();

    type::vector<sofa::type::Vec3>& my_positions = *(output->seqPoints.beginEdit());
    type::vector<Triangle >& my_triangles = *(output->seqTriangles.beginEdit());
    type::vector<sofa::type::Vec2>& my_uv = *(output->seqUVs.beginEdit());

    my_positions.clear();
    my_triangles.clear();
    my_uv.clear();

    const type::vector<type::fixed_array <unsigned int,3> >& my_indicesComponents = input->_indicesComponents.getValue();

    unsigned int my_numberShape = shapeNumber.getValue();

    unsigned int beginIdx = 0;
    unsigned int endIdx   = 0;
    if (my_numberShape >= my_indicesComponents.size())
    {
        msg_error() << "Number of the shape not valid";
    }
    else
    {
        unsigned int numNodes = 0, numTriangles = 0;
        for (unsigned int i=0; i<my_indicesComponents.size(); ++i)
        {
            if (my_indicesComponents[i][0] == my_numberShape)
            {
                endIdx = beginIdx + my_indicesComponents[i][2];
                if(positionsI.size()>0 )
                {
                    for (unsigned int j=0; j<my_indicesComponents[i][1]; ++j)
                    {
                        my_positions.push_back(positionsI[j+numNodes]);
                        my_uv.push_back(uvI[j+numNodes]);
                    }
                }

                if(trianglesI.size() > 0 )
                {
                    for (unsigned int j=0; j<my_indicesComponents[i][2]; ++j)
                    {
                        Triangle triangleTemp(trianglesI[j+numTriangles][0]-numNodes, trianglesI[j+numTriangles][1]-numNodes, trianglesI[j+numTriangles][2]-numNodes);
                        my_triangles.push_back(triangleTemp);
                    }
                }

                break;
            }
            numNodes += my_indicesComponents[i][1];
            numTriangles += my_indicesComponents[i][2];
            beginIdx     += numTriangles;
        }
    }

    indexBegin.setValue(beginIdx);
    indexEnd.setValue(endIdx);

    output->seqPoints.endEdit();
    output->seqTriangles.endEdit();
    output->seqUVs.endEdit();

}


} // namespace engine

} // namespace component

} // namespace sofa
