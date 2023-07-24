python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/left_frames" --data_config config_endovis18_test.yml  --model_config model_biastuning.yml --save_path "./samzs_results/seq1" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/labels" --labels_of_interest "ultrasound probe"

echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/left_frames" --data_config config_endovis18_test.yml  --model_config model_biastuning.yml --save_path "./samzs_results/seq2" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/labels" --labels_of_interest "ultrasound probe"


echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/left_frames" --data_config config_endovis18_test.yml  --model_config model_biastuning.yml --save_path "./samzs_results/seq3" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/labels" --labels_of_interest "ultrasound probe"


echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/left_frames" --data_config config_endovis18_test.yml  --model_config model_biastuning.yml --save_path "./samzs_results/seq4" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/labels" --labels_of_interest "ultrasound probe"


echo "......................."