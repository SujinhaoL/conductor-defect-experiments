#include <windows.h>
#include <iostream>
static_assert(_MSC_VER == 1929, "Expected MSVC v142, compiler version 19.29");
static_assert(sizeof(void*) == 8, "Expected a 64-bit target");
int main() {
    std::cout << "MSVC_VERSION=" << _MSC_VER << "\n";
    std::cout << "WINDOWS_SDK_LINK_PASS=" << (GetCurrentProcessId() != 0) << "\n";
    std::cout << "STEP02A_CPP_COMPILE_RUN_PASS\n";
    return 0;
}
