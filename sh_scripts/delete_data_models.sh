echo $PWD
find $PWD/data/data_split/ -type f -name "train.csv" -delete
find $PWD/data/data_split/ -type f -name "val.csv" -delete
find $PWD/data/data_split/ -type f -name "test.csv" -delete
find $PWD/data/data_split/ -type f -name "feat_target_test.pkl" -delete
find $PWD/data/data_split/ -type f -name "feat_target_train.pkl" -delete
find $PWD/data/data_split/ -type f -name "feat_target_val.pkl" -delete
find $PWD/data/data_split/ -type f -name "le.pkl" -delete
find $PWD/data/ -type f -name "train.csv" -delete
find $PWD/data/ -type f -name "val.csv" -delete
find $PWD/data/ -type f -name "test.csv" -delete
find $PWD/plots/ -type f -name "crnn_plot.csv" -delete
find $PWD/plots/ -type f -name "rnn_plot.csv" -delete
find $PWD/plots/ -type f -name "trainer_state.json" -delete
find $PWD/test_preds/ -type f -name "predict_crnn.pkl" -delete
find $PWD/test_preds/ -type f -name "predict_rnn.pkl" -delete
find $PWD/test_preds/ -type f -name "roberta_preds.pkl" -delete
find $PWD/model/tf_nn_arch/ -type f -name "emb_dim.pkl" -delete
find $PWD/model/tf_nn_arch/ -type f -name "emb_mtrx.pkl" -delete
find $PWD/model/tf_nn_arch/ -type f -name "max_len.pkl" -delete
find $PWD/model/tf_nn_arch/ -type f -name "num_o_words.pkl" -delete
find $PWD/model/tf_nn_arch/ -type f -name "tf_tok.json" -delete
find $PWD/model/ -type f -name "model_crnn.h5" -delete
find $PWD/model/ -type f -name "model_rnn.h5" -delete
find $PWD/model/ -type d -name "roberta_model" -exec rm -rf {} \;
find $PWD/model/ -type d -name "roberta_text_cls" -exec rm -rf {} \;
find $PWD/model/ -type d -name "roberta_tok" -exec rm -rf {} \;
find $PWD/plots/ -type d -name "logging_dir" -exec rm -rf {} \;