python generate_predictions_baselines.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/left_frames" --data_config config_endovis18_test.yml  --model_config model_baseline.yml --save_path "./unet_results/seq1" --pretrained_path ../../unet_ev18.pth --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/labels"

echo "......................."

python generate_predictions_baselines.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/left_frames" --data_config config_endovis18_test.yml  --model_config model_baseline.yml --save_path "./unet_results/seq2" --pretrained_path ../../unet_ev18.pth --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/labels"


echo "......................."

python generate_predictions_baselines.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/left_frames" --data_config config_endovis18_test.yml  --model_config model_baseline.yml --save_path "./unet_results/seq3" --pretrained_path ../../unet_ev18.pth --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/labels"


echo "......................."

python generate_predictions_baselines.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/left_frames" --data_config config_endovis18_test.yml  --model_config model_baseline.yml --save_path "./unet_results/seq4" --pretrained_path ../../unet_ev18.pth --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/labels"


echo "......................."