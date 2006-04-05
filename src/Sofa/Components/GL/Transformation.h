#ifndef SOFA_COMPONENTS_GL_TRANSFORMATION_H
#define SOFA_COMPONENTS_GL_TRANSFORMATION_H

namespace Sofa
{

namespace Components
{

namespace GL
{


class   		Transformation
{
public:

    double			translation[3];
    double			scale[3];
    double			rotation[4][4];

    double			objectCenter[3];

private:

public:

    Transformation();	// constructor
    ~Transformation();	// destructor



    Transformation&	operator=(const Transformation& transform);

    void			Apply();
    void			ApplyWithCentring();
    void			ApplyInverse();

private:void		InvertTransRotMatrix(double matrix[4][4]);
    void			InvertTransRotMatrix(double sMatrix[4][4],
            double dMatrix[4][4]);
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif // __TRANSFORMATION_H__
