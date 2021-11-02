expe_name="libri460_fast_vq_spkdelta_l2norm"
template="libri460_fast_vq_spkdelta"

rsync -av --progress "$template/" "$expe_name/" --exclude "log" --exclude "model" --exclude "slurm*"

mkdir -p "$expe_name/log"
mkdir -p "$expe_name/model"
