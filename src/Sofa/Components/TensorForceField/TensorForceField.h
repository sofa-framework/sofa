#ifndef _TENSORMASSFORCEFIELD_H_
#define _TENSORMASSFORCEFIELD_H_


#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Sofa/Abstract/VisualModel.h"


namespace Sofa
{


namespace Components
{



class VertexTensor
{
public:
    double tensor[3][3];

    void resetToNull();

}; //class VertexTensor



class Edge
{
public:
    int index;

    int vertex[2];

}; // class Edge



class EdgeTensor
{
public:
    double tensor[3][3];

    void resetToNull();

}; // class EdgeTensor



class Triangle
{
public:
    int index;

    int vertex[3];

    double shapeVector[3];
    double squareRestArea;

}; // class Triangle



class Tetrahedron
{
public:
    int index;

    int vertex  [4];
    int edge    [6];
    int triangle[4];

    double triangleShapeVector[4][3];
    double restVolume;

}; // class Tetrahedron



/**
 * @class TensorForceField
 * @brief Tensors based internal force field class.
 *
 * Internal Force field (ie it has only one parent object) based on a Finite
 * Element Model modelisation, using strain and stiffness tensors on vertices
 * and edges.
 */
template <class DataTypes>
class TensorForceField : public Core::ForceField,
    public Abstract::VisualModel
{
public:


    TensorForceField (const char *filename);


    TensorForceField (Core::MechanicalObject<DataTypes>* object,
            const char* filename);


    virtual void addForce ();


    virtual void addDForce();

    // -- VisualModel interface
    void draw();

    // -- VisualModel interface
    void initTextures();

    // -- VisualModel interface
    void update();


public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

private:
    // Mechanical Object containing this Force Field
    Core::MechanicalObject<DataTypes>* object_;
    // damping factor
    double alpha_;
    // Lame coefficient
    double lambda_, mu_;
    std::vector< Coord >    vertex_      ;
    std::vector< Edge >         edge_        ;
    std::vector< Triangle >     triangle_    ;
    std::vector< Tetrahedron >  tetrahedron_ ;
    std::vector< VertexTensor > vertexTensor_;
    std::vector< EdgeTensor >   edgeTensor_  ;

private:
    // load the topological informations from a file.
    void load(const char *filename);

    // initialize the tensors
    void initialize();

    // search for the edge connecting the given vertices, create it if not found
    int getEdge(const int v0, const int v1);

    // search for the triangle connecting the given vertices, create it if not found
    int getTriangle(const int v0, const int v1, const int v2);

    // add the elastic tensors for this tetrahedron
    void addElasticTensors(Tetrahedron& tetra);

}; // class TensorForceField



} // namespace Components



} // namespace Sofa



#endif /* _TENSORMASSFORCEFIELD_H_ */
