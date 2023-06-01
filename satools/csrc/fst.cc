#include "fst.h"

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst &fst) {
    try {
      fst::ReadFstKaldi(rxfilename, &fst);
    } catch (...) {
        std::cerr << "Error opening " << rxfilename << std::endl;
        return;
    }
    return;
}

void Project(
        fst::StdVectorFst &fst,
        bool project_output
        ) {
    fst::Project(&fst, project_output ? fst::PROJECT_OUTPUT : fst::PROJECT_INPUT);
}

