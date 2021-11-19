expe_name="libri460_fast_vq_idonly"
template="libri460_fast_vq"

rsync -av --progress "$template/" "$expe_name/" --exclude "log" --exclude "model" --exclude "slurm*"

mkdir -p "$expe_name/log"
mkdir -p "$expe_name/model"
