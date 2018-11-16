if [ ! -d ./graph ]; then
  mkdir -p ./graph;
fi

mvNCCompile -s 12 ./logs/frozen_model.pb -in=input -on=output -o='./graph/graph'
