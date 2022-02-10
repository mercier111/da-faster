import os

net = "res101"
part = "test_t"
dataset = "clipart"
begin_epoch = 8
end_epoch = 12

model_prefix = "experiments/DA_Faster/clipart/model/clipart_"

commond = "python eval/test.py --net {}  --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")