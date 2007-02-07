#ifndef SOFA_HELPER_GL_TRANSFORMATION_H
#define SOFA_HELPER_GL_TRANSFORMATION_H

namespace sofa
{

namespace helper
{

namespace gl
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

    template<class Vector>
    Vector operator*(Vector v) const
    {
        for(int c=0; c<3; c++)
            v[c] *= scale[c];
        Vector r;
        for(int c=0; c<3; c++)
            r[c] = rotation[0][c]*v[0]+rotation[1][c]*v[1]+rotation[2][c]*v[2];
        for(int c=0; c<3; c++)
            r[c] += translation[c];
        return r;
    }

private:void		InvertTransRotMatrix(double matrix[4][4]);
    void			InvertTransRotMatrix(double sMatrix[4][4],
            double dMatrix[4][4]);
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif // __TRANSFORMATION_H__
