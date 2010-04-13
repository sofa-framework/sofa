



__kernel void memset(
    __global unsigned int *res1,
    unsigned int offset,
    unsigned int value
)
{

    int index = get_global_id(0)*4 + offset;

    res1[index] = value;
    res1[index+1] = value;
    res1[index+2] = value;
    res1[index+3] = value;


}

