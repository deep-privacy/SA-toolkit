#include "decoder.h"

kaldi::LatticeFasterDecoderConfig CreateLatticeFasterDecoderConfig(float beam, int32 max_active, int32 min_active, float lattice_beam) { 
    kaldi::LatticeFasterDecoderConfig  opts; 
    opts.beam = beam; 
    opts.max_active = max_active; 
    opts.min_active = min_active; 
    opts.lattice_beam = lattice_beam; 
    return opts; 
}

std::tuple<std::string, std::vector<int32>, std::vector<int32>, kaldi::CompactLattice> MyDecodeUtteranceLatticeFaster(
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
    kaldi::CompactLattice clat;

    if (!decoder.Decode(&decodable)) {
        std::cout << "Failed to decode utterance " << std::endl;
        return std::make_tuple("", words, alignment, clat);
    }
    if (!decoder.ReachedFinal()) {
        if (allow_partial) {
            std::cout << "Outputting partial output for utterance "
                << " since no final-state reached\n";
        } else {
            std::cout << "Not producing output for utterance "
                << " since no final-state reached and "
                << "allow_partial=False.\n";
            return std::make_tuple("", words, alignment, clat);
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
            std::cout << "Failed to get traceback for utterance " << std::endl;
            exit(1);
        }

        fst::GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
        num_frames = alignment.size();
        if (word_syms != NULL) {
            for (size_t i = 0; i < words.size(); i++) {
                std::string c = word_syms->Find(words[i]);
                if (c == "") {
                    std::cout << "Word-id " << words[i] << " not in symbol table." << std::endl;
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


    kaldi::Lattice lat;
    decoder.GetRawLattice(&lat);
    if (lat.NumStates() == 0) {
        std::cout << "Unexpected problem getting lattice for utterance " << std::endl;
        exit(1);
    }
    if(!fst::DeterminizeLatticePhonePrunedWrapper(
                trans_model,
                &lat,
                decoder.GetOptions().lattice_beam,
                &clat,
                decoder.GetOptions().det_opts)) {
        std::cout << "Determinization finished earlier than the beam for utterance" << std::endl;
    }

    if (acoustic_scale != 0.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);

    return std::make_tuple(s, words, alignment, clat);
}

std::tuple<std::string, std::vector<int32>, std::vector<int32>, kaldi::CompactLattice> MappedLatticeFasterRecognizer(
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
                << word_syms_filename  << std::endl;
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


kaldi::CompactLattice LatticeLmrescore(
        fst::VectorFst<fst::StdArc>  &std_lm_fst_in,
        kaldi::CompactLattice &clat,
        float lm_scale_
        ){

    BaseFloat lm_scale = lm_scale_;
    int32 num_states_cache = 50000;

    fst::VectorFst<fst::StdArc> *std_lm_fst = &std_lm_fst_in;


    if (std_lm_fst->Properties(fst::kILabelSorted, true) == 0) {
        // Make sure LM is sorted on ilabel.
        fst::ILabelCompare<fst::StdArc> ilabel_comp;
        fst::ArcSort(std_lm_fst, ilabel_comp);
    }


    // mapped_fst is the LM fst interpreted using the LatticeWeight semiring,
    // with all the cost on the first member of the pair (since it's a graph
    // weight).
    fst::CacheOptions cache_opts(true, num_states_cache);
    fst::MapFstOptions mapfst_opts(cache_opts);
    fst::StdToLatticeMapper<BaseFloat> mapper;
    fst::MapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<BaseFloat> >
        lm_fst(*std_lm_fst, mapper, mapfst_opts);

    // The next fifteen or so lines are a kind of optimization and
    // can be ignored if you just want to understand what is going on.
    // Change the options for TableCompose to match the input
    // (because it's the arcs of the LM FST we want to do lookup
    // on).
    fst::TableComposeOptions compose_opts(fst::TableMatcherOptions(),
            true, fst::SEQUENCE_FILTER,
            fst::MATCH_INPUT);

    // The following is an optimization for the TableCompose
    // composition: it stores certain tables that enable fast
    // lookup of arcs during composition.
    fst::TableComposeCache<fst::Fst<kaldi::LatticeArc> > lm_compose_cache(compose_opts);


    // Convert the CompactLattice to a Lattice.
    kaldi::Lattice lat;
    ConvertLattice(clat, &lat);


    if (lm_scale != 0.0) {
        // Only need to modify it if LM scale nonzero.
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &lat);
        fst::ArcSort(&lat, fst::OLabelCompare<kaldi::LatticeArc>());

        kaldi::Lattice composed_lat;
        // Could just do, more simply: Compose(lat, lm_fst, &composed_lat);
        // and not have lm_compose_cache at all.
        // The command below is faster, though; it's constant not
        // logarithmic in vocab size.
        fst::TableCompose(lat, lm_fst, &composed_lat, &lm_compose_cache);

        fst::Invert(&composed_lat); // make it so word labels are on the input.
        kaldi::CompactLattice determinized_lat;
        fst::DeterminizeLattice(composed_lat, &determinized_lat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_lat);
        return determinized_lat;
    } else {
        // zero scale so nothing to do.
        kaldi::CompactLattice compact_lat;
        ConvertLattice(lat, &compact_lat);
        return compact_lat;
    }

}

kaldi::CompactLattice LatticeLmrescoreConstArpa(
        std::string  &constarpalm_filename,
        kaldi::CompactLattice &clat,
        float lm_scale_
        ){

    BaseFloat lm_scale = lm_scale_;

    // Reads the language model in ConstArpaLm format.
    kaldi::ConstArpaLm const_arpa;
    kaldi::ReadKaldiObject(constarpalm_filename, &const_arpa);


    if (lm_scale != 0.0) {
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0/lm_scale), &clat);
        fst::ArcSort(&clat, fst::OLabelCompare<kaldi::CompactLatticeArc>());

        // Wraps the ConstArpaLm format language model into FST. We re-create it
        // for each lattice to prevent memory usage increasing with time.
        kaldi::ConstArpaLmDeterministicFst const_arpa_fst(const_arpa);

        // Composes lattice with language model.
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticeDeterministic(clat,
                &const_arpa_fst, &composed_clat);

        // Determinizes the composed lattice.
        kaldi::Lattice composed_lat;
        ConvertLattice(composed_clat, &composed_lat);
        fst::Invert(&composed_lat);
        kaldi::CompactLattice determinized_clat;
        fst::DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_clat);
        return determinized_clat;
    } else {
        return clat;
    }

}


std::tuple<std::string, std::vector<int32>, std::vector<int32>, kaldi::CompactLattice> LatticeBestPath(
        std::string &trans_model_s,
        std::string &word_syms_filename,
        kaldi::CompactLattice &clat,
        float acoustic_scale_,
        float lm_scale_
        ){

    BaseFloat acoustic_scale = acoustic_scale_;
    BaseFloat lm_scale = lm_scale_;

    std::string s;
    std::vector<int32> alignment;
    std::vector<int32> words;
    kaldi::LatticeWeight weight;


    kaldi::TransitionModel trans_model;
    kaldi::ReadKaldiObject(trans_model_s, &trans_model);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
        if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
            std::cout << "Could not read symbol table from file "
                << word_syms_filename  << std::endl;
            exit(1);
        }
    }

    fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
    kaldi::CompactLattice clat_best_path;
    kaldi::CompactLatticeShortestPath(clat, &clat_best_path);  // A specialized
    kaldi::Lattice best_path;
    ConvertLattice(clat_best_path, &best_path);

    fst::GetLinearSymbolSequence(best_path, &alignment, &words, &weight);

    if (word_syms != NULL) {
        for (size_t i = 0; i < words.size(); i++) {
            std::string c = word_syms->Find(words[i]);
            if (c == "") {
                std::cout << "Word-id " << words[i] << " not in symbol table." << std::endl;
                exit(1);
            }
            s += c;
            s += ' ';
        }
        std::cerr << '\n';
    }

    return std::make_tuple(s, words, alignment, clat_best_path);

}

kaldi::CompactLattice LatticeAlignWordsLexicon(
        std::string &trans_model_s,
        std::string &align_lexicon_rxfilename,
        kaldi::CompactLattice &clat
        ){

    kaldi::WordAlignLatticeLexiconOpts opts;


    std::vector<std::vector<int32> > lexicon;
    {
        bool binary_in;
        kaldi::Input ki(align_lexicon_rxfilename, &binary_in);
        if (!kaldi::ReadLexiconForWordAlign(ki.Stream(), &lexicon)) {
            std::cout << "Error reading alignment lexicon from "
                << align_lexicon_rxfilename << std::endl;
            exit(1);
        }
    }

    kaldi::TransitionModel tmodel;
    kaldi::ReadKaldiObject(trans_model_s, &tmodel);

    kaldi::WordAlignLatticeLexiconInfo lexicon_info(lexicon);
    { std::vector<std::vector<int32> > temp; lexicon.swap(temp); }
    // No longer needed.

    kaldi::CompactLattice aligned_clat;

     bool ok = kaldi::WordAlignLatticeLexicon(clat, tmodel, lexicon_info, opts,
                                        &aligned_clat);
     if (!ok) {
         std::cout << "Error creating aligment lattice " << std::endl;
         exit(1);
     }

     if (aligned_clat.Start() == fst::kNoStateId) {
         std::cout << "Aligment lattice is empty" << std::endl;
     }
     kaldi::TopSortCompactLatticeIfNeeded(&aligned_clat);
     return aligned_clat;
}

std::tuple<std::vector<int32>, std::vector<float>,std::vector<float>, std::vector<std::string>> NbestToCTM(
        std::string &word_syms_filename,
        kaldi::CompactLattice &clat,
        float frame_shift_,
        bool print_silence
        ) {

    BaseFloat frame_shift = frame_shift_;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
        if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
            std::cout << "Could not read symbol table from file "
                << word_syms_filename  << std::endl;
            exit(1);
        }
    }

    std::vector<int32> words, times, lengths, words_out;
    std::vector<float> times_out, lengths_out;
    std::vector<std::string> wordstxt;
    if (!kaldi::CompactLatticeToWordAlignment(clat, &words, &times, &lengths)) {
         std::cout << "NbestToCTM Format conversion failed " << std::endl;
         exit(1);
    } else {
        for (size_t i = 0; i < words.size(); i++) {
            if (words[i] == 0 && !print_silence)  // Don't output anything for <eps> links, which
                continue; // correspond to silence....

            times_out.push_back(frame_shift * times[i]);
            lengths_out.push_back(frame_shift * lengths[i]);
            words_out.push_back(words[i]);

            if (word_syms != NULL) {
                std::string c = word_syms->Find(words[i]);
                if (c == "") {
                    std::cout << "Word-id " << words[i] << " not in symbol table." << std::endl;
                    exit(1);
                }
                wordstxt.push_back(c);

            }
        }
    }
    return std::make_tuple(words_out, times_out, lengths_out, wordstxt);
}
