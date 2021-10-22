expe_name="libri460_fast2"
template="libri460_fast"

rsync -av --progress "$template/" "$expe_name/" --exclude "log" --exclude "model" --exclude "slurm*"

mkdir -p "$expe_name/log"
mkdir -p "$expe_name/model"
