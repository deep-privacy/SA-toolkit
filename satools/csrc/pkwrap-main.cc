#include "pkwrap-main.h"

static bool instantiated = false;
inline void InstantiateKaldiCuda() {
    if(!instantiated) {
        kaldi::CuDevice::Instantiate().SelectGpuId("yes");
        kaldi::CuDevice::Instantiate().AllowMultithreading();
        instantiated = true;
    }
}
