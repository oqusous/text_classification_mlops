find ./data/data_split/ -type f -name "train.csv" -delete
find ./data/data_split/ -type f -name "val.csv" -delete
find ./data/data_split/ -type f -name "test.csv" -delete
find ./data/data_split/ -type f -name "feat_target_test.pkl" -delete
find ./data/data_split/ -type f -name "feat_target_train.pkl" -delete
find ./data/data_split/ -type f -name "feat_target_val.pkl" -delete
find ./data/data_split/ -type f -name "le.pkl" -delete
find ./data/ -type f -name "train.csv" -delete
find ./data/ -type f -name "val.csv" -delete
find ./data/ -type f -name "test.csv" -delete
find ./plots/ -type f -name "crnn_plot.csv" -delete
find ./plots/ -type f -name "rnn_plot.csv" -delete
find ./plots/ -type f -name "trainer_state.json" -delete
find ./test_preds/ -type f -name "predict_crnn.pkl" -delete
find ./test_preds/ -type f -name "predict_rnn.pkl" -delete
find ./test_preds/ -type f -name "roberta_preds.pkl" -delete
find ./model/tf_nn_arch/ -type f -name "emb_dim.pkl" -delete
find ./model/tf_nn_arch/ -type f -name "emb_mtrx.pkl" -delete
find ./model/tf_nn_arch/ -type f -name "max_len.pkl" -delete
find ./model/tf_nn_arch/ -type f -name "num_o_words.pkl" -delete
find ./model/tf_nn_arch/ -type f -name "tf_tok.json" -delete
find ./model/ -type f -name "model_crnn.h5" -delete
find ./model/ -type f -name "model_rnn.h5" -delete
find ./model/ -type d -name "roberta_model" -exec rm -rf {} \;
find ./model/ -type d -name "roberta_text_cls" -exec rm -rf {} \;
find ./model/ -type d -name "roberta_tok" -exec rm -rf {} \;