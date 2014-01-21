#
# The script defines the following variables:
#
##############################################################################
# Cutil library: CUDA_SPARSE_LIBRARY
##############################################################################
#
#
# CUDA_TOOLKIT_ROOT_DIR -- Path to the CUDA toolkit. As the sparse library is
# part of the CUDA toolkit, this suffices to find the
# library.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindCUDASparse.cmake
find_library(CUDA_SPARSE_LIBRARY
    NAMES cusparse
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
    PATH_SUFFIXES "/lib64" "/lib" "/lib/x86_64-linux-gnu" "lib/x64" "lib/Win32"
    DOC "Location of sparse library"
    NO_DEFAULT_PATH
    )

mark_as_advanced(CUDA_SPARSE_LIBRARY)

#############################
# Check for required components
if(CUDA_SPARSE_LIBRARY)
    set(CUDASparse_FOUND TRUE)
endif(CUDA_SPARSE_LIBRARY)
