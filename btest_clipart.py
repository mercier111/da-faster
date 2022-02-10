import os

net = "res101"
part = "test_t"
dataset = "clipart"

begin_epoch = 5
end_epoch = 5

model_prefix = "experiments/cycle_DA_Faster_new_gen_5/clipart/model/clipart_"

commond = "python -u eval/btest.py --net {} --cuda --cag  --source_bayes 0 --target_bayes 0 --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")