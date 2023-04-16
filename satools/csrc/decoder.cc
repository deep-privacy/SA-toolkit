#include "decoder.h"

kaldi::LatticeFasterDecoderConfig CreateLatticeFasterDecoderConfig(float beam, int32 max_active, int32 min_active, float lattice_beam) { 
    kaldi::LatticeFasterDecoderConfig  opts; 
    opts.beam = beam; 
    opts.max_active = max_active; 
    opts.min_active = min_active; 
    opts.lattice_beam = lattice_beam; 
    return opts; 
}

std::tuple<std::string, std::vector<int32>, std::vector<int32>> MyDecodeUtteranceLatticeFaster(
        kaldi::LatticeFasterDecoderTpl<fst::Fst<fst::StdArc> > &decoder,
        kaldi::DecodableInterface &decodable, // not const but is really an input.
        const kaldi::TransitionInformation &trans_model,
        const fst::SymbolTable *word_syms,
        double acoustic_scale,
        bool allow_partial,
        double *like_ptr) { // puts utterance's like in like_ptr on success.
    using fst::VectorFst;

    std::vector<int32> alignment;
    std::vector<int32> words;

    if (!decoder.Decode(&decodable)) {
        std::cout << "Failed to decode utterance ";
        return std::make_tuple("", words, alignment);
    }
    if (!decoder.ReachedFinal()) {
        if (allow_partial) {
            std::cout << "Outputting partial output for utterance "
                << " since no final-state reached\n";
        } else {
            std::cout << "Not producing output for utterance "
                << " since no final-state reached and "
                << "allow_partial=False.\n";
            return std::make_tuple("", words, alignment);
        }
    }

    double likelihood;
    kaldi::LatticeWeight weight;
    int32 num_frames;
    std::string s;

    { // First do some stuff with word-level traceback...
        VectorFst<kaldi::LatticeArc> decoded;
        if (!decoder.GetBestPath(&decoded)){
            // Shouldn't really reach this point as already checked success.
            std::cout << "Failed to get traceback for utterance ";
            exit(1);
        }

        fst::GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
        num_frames = alignment.size();
        if (word_syms != NULL) {
            for (size_t i = 0; i < words.size(); i++) {
                std::string c = word_syms->Find(words[i]);
                if (c == "") {
                    std::cout << "Word-id " << words[i] << " not in symbol table.";
                    exit(1);
                }
                s += c;
                s += ' ';
            }
            std::cerr << '\n';
        }
        likelihood = -(weight.Value1() + weight.Value2());
    }

    *like_ptr = likelihood;
    return std::make_tuple(s, words, alignment);
}

std::tuple<std::string, std::vector<int32>, std::vector<int32>> MappedLatticeFasterRecognizer(
        kaldi::Matrix<kaldi::BaseFloat> loglikes,
        kaldi::LatticeFasterDecoderConfig &config,
        std::string &trans_model_s,
        std::string &fst_in_str,
        std::string &word_syms_filename,
        float acoustic_scale,
        bool allow_partial
        ){

    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    if (loglikes.NumRows() == 0) {
        std::cout << "Zero-length utterance" << std::endl;
        exit(1);
    }

    kaldi::TransitionModel trans_model;
    kaldi::ReadKaldiObject(trans_model_s, &trans_model);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
        if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
            std::cout << "Could not read symbol table from file "
                << word_syms_filename;
            exit(1);
        }
    }

    if (kaldi::ClassifyRspecifier(fst_in_str, NULL, NULL) == kaldi::kNoRspecifier) {

        // Input FST is just one FST, not a table of FSTs.
        Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

        {
            kaldi::LatticeFasterDecoder decoder(*decode_fst, config);
            kaldi::DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

            double like;
            return MyDecodeUtteranceLatticeFaster(
                    decoder, decodable, trans_model, word_syms,
                    acoustic_scale, allow_partial, &like);
        }
        delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
        kaldi::SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
        kaldi::LatticeFasterDecoder decoder(fst_reader.Value(), config);
        kaldi::DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
        double like;
        return MyDecodeUtteranceLatticeFaster(
                decoder, decodable, trans_model, word_syms, acoustic_scale,
                allow_partial, &like);
    }

}
