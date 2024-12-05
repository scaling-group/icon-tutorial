
python3 -m src.train
# if you want to change the configuration, you can pass the argument like this:
# python3 -m src.train acc=conserve # change the acc config file to conserve.yaml
# python3 -m src.train model.pe=no # change model.pe to "no"
# python3 -m src.train board=True # to enable boarding to WandB

echo "Done"
