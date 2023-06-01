#ifndef PKWRAP_FST_H_
#define PKWRAP_FST_H_
#include "common.h"
#include "fstext/fstext-lib.h"

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst &fst);

void Project(
        fst::StdVectorFst &fst,
        bool project_output
        );

#endif
