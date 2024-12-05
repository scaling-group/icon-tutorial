dir=./scratch/data/data_weno_cubic

if [ ! -d "$dir" ]; then
  mkdir -p "$dir"
fi

traineqns=1000 # training.
trainnum=100 # number of cond-qoi pairs per equation
valideqns=10 # validation, only for visualization during training
validnum=100 # number of cond-qoi pairs per equation

python3 -m src.datagen.generate --dir $dir --name valid --eqns $valideqns --num $validnum --file_split 1  --truncate 100 --seed 101 && 
python3 -m src.datagen.generate --dir $dir --name train --eqns $traineqns --num $trainnum --file_split 10 --truncate 100 --seed 1 &&

echo "Done"
