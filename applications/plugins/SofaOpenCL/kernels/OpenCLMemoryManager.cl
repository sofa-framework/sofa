
__kernel void MemoryManager_memset(
    __global unsigned int *res1,
    unsigned int offset,
    unsigned int value
)
{

    int index = get_global_id(0) + offset;

    res1[index] = value;
}

