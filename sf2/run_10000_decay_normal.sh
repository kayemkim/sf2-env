python small_network.py \
    --rom_path '/root/sf2-workspace/sf2-env/StreetFighterIISpecialChampionEdition-Genesis' \
    --model_path '/root/sf2-workspace/sf2-env/sf2/models/64_nodes_decay_normal_10000_ep/' \
    --scenario 'scenario' \
    --stack_size 5 \
     --learning_rate 0.005 \
    --total_episodes 15000 \
    --batch_size 64 \
    --explore_start 1.0 \
    --explore_stop 0.01 \
    --decay_rate 0.00001 \
    --gamma 0.95 \
    --memory_size 100000
