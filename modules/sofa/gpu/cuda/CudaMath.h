#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <cuda_runtime.h>

__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ void operator+=(float3& a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

__device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ void operator-=(float3& a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ void operator-=(float3& a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__device__ float3 operator-(float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__ void operator*=(float3& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__device__ float3 operator*(float a, float3 b)
{
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 make_float3(float4 f)
{
    return make_float3(f.x,f.y,f.z);
}

__device__ float4 make_float4(float3 f, float w=0.0)
{
    return make_float4(f.x,f.y,f.z,w);
}


__device__ float3 mul(float3 a, float3 b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

__device__ float norm2(float3 a)
{
    return a.x*a.x+a.y*a.y+a.z*a.z;
}

__device__ float norm(float3 a)
{
    return sqrtf(norm2(a));
}

__device__ float invnorm(float3 a)
{
    return rsqrtf(norm2(a));
}

class /*__align__(4)*/ matrix3
{
public:
    float3 x,y,z;
    /*
        float3 getLx() { return x; }
        float3 getLy() { return y; }
        float3 getLz() { return z; }

        float3 getCx() { return make_float3(x.x,y.x,z.x); }
        float3 getCy() { return make_float3(x.y,y.y,z.y); }
        float3 getCz() { return make_float3(x.z,y.z,z.z); }

        void setLx(float3 v) { x = v; }
        void setLy(float3 v) { y = v; }
        void setLz(float3 v) { z = v; }

        void setCx(float3 v) { x.x = v.x; y.x = v.y; z.x = v.z; }
        void setCy(float3 v) { x.y = v.x; y.y = v.y; z.y = v.z; }
        void setCz(float3 v) { x.z = v.x; y.z = v.y; z.z = v.z; }
    */
    __device__ float3 operator*(float3 v)
    {
        return make_float3(dot(x,v),dot(y,v),dot(z,v));
    }
    __device__ float3 mulT(float3 v)
    {
        return x*v.x+y*v.y+z*v.z;
    }
    __device__ float mulX(float3 v)
    {
        return dot(x,v);
    }
    __device__ float mulY(float3 v)
    {
        return dot(y,v);
    }
    __device__ float mulZ(float3 v)
    {
        return dot(z,v);
    }
    __device__ void readAoS(const float* data)
    {
        x.x=*data; data+=blockDim.x;
        x.y=*data; data+=blockDim.x;
        x.z=*data; data+=blockDim.x;
        y.x=*data; data+=blockDim.x;
        y.y=*data; data+=blockDim.x;
        y.z=*data; data+=blockDim.x;
        z.x=*data; data+=blockDim.x;
        z.y=*data; data+=blockDim.x;
        z.z=*data; data+=blockDim.x;
    }
    __device__ void writeAoS(float* data)
    {
        *data=x.x; data+=blockDim.x;
        *data=x.y; data+=blockDim.x;
        *data=x.z; data+=blockDim.x;
        *data=y.x; data+=blockDim.x;
        *data=y.y; data+=blockDim.x;
        *data=y.z; data+=blockDim.x;
        *data=z.x; data+=blockDim.x;
        *data=z.y; data+=blockDim.x;
        *data=z.z; data+=blockDim.x;
    }
};

#endif
