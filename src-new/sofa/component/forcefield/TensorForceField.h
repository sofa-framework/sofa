#ifndef SOFA_COMPONENT_FORCEFIELD_TENSORFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TENSORFORCEFIELD_H


#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

/**
 * @class TensorForceField
 * @brief Tensors based internal force field class.
 *
 * Internal Force field (ie it has only one parent object) based on a Finite
 * Element Model modelisation, using strain and stiffness tensors on vertices
 * and edges.
 */
template <class DataTypes>
class TensorForceField : public core::componentmodel::behavior::BaseForceField,
    public core::VisualModel
{
public:

    TensorForceField (const char *filename);


    TensorForceField (component::MechanicalObject<DataTypes>* object,
            const char* filename);


    virtual void addForce ();


    virtual void addDForce();

    virtual double getPotentialEnergy();

    // -- Temporary added here for matrix ForceField
    void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};
    void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};
    void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
    void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int & ) {};

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
    class VertexTensor
    {
    public:
        Real tensor[3][3];

        void resetToNull()
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    tensor[i][j] = 0.0;
                }
            }
        }

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
        Real tensor[3][3];

        void resetToNull()
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    tensor[i][j] = 0.0;
                }
            }
        }

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

        Real triangleShapeVector[4][3];
        Real restVolume;

    }; // class Tetrahedron


private:
    // Mechanical Object containing this Force Field
    component::MechanicalObject<DataTypes>* object_;
    // damping factor
    Real alpha_;
    // Lame coefficient
    Real lambda_, mu_;
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



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif /* _TENSORMASSFORCEFIELD_H_ */
