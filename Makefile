DATA_PATH=./data
pretrain_dd_config1:
	# when running with deepspeed, learning rate must be provided through config.json
	deepspeed pretrain_deepspeed.py --batch_size=99 --d_model=256 --dataset=dd --encoder_heads=8 --encoder_layers=6 --epochs 20 --log_interval=50 --projection=128 --tracking=0 --val_interval=100 --deepspeed --deepspeed_config config.json
setup_mutual:
	echo "DATA_PATH = $(DATA_PATH)"
	git clone https://github.com/Nealcly/MuTual.git $(DATA_PATH)/mutual
