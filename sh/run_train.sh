python trainer.py  --input ./train_data/ctb6/ctb6.train ./train_data/ctb6/ctb6.train.label --vocabulary vocab.char.txt vocab.label.txt --model lstm_cnn --validation ./train_data/ctb6/ctb6.dev --references ./train_data/ctb6/ctb6.dev.label --parameters=batch_size=256,device_list=[1],train_steps=100000000

