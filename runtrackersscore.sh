echo "start trackersscore"
logdir=trackersscore
if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi
python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_10 --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_object --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_object.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_goal --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_goal.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_spatial --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_spatial.txt 2>&1 &
sleep 3h


echo "start trackersscore heavy"
logdir=trackersscoreheavy
if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi
python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_10 --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_object --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_object.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_goal --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_goal.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_spatial --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_spatial.txt 2>&1 &
sleep 3h

#各再跑一次到新文件夹
echo "start trackersscore new"
logdir=trackersscorenew
if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi
python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_10 --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_object --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_object.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_goal --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_goal.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python evalpi05_adaptivefine.py --config_path evalpi05.yaml --env_task=libero_spatial --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_spatial.txt 2>&1 &
sleep 3h


echo "start trackersscore heavy new"
logdir=trackersscoreheavynew
if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi
python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_10 --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_object --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_object.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_goal --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_goal.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python evalpi05_adaptivefine_heavy.py --config_path evalpi05.yaml --env_task=libero_spatial --inference_overlap_steps=0 --output_dir=testadapt --use_adaptive_chunk=True > $logdir/libero_spatial.txt 2>&1 &
sleep 3h


