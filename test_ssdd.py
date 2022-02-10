import os

net = "res101"
part = "test_t"
dataset = "LEVIR"
begin_epoch = 15
end_epoch = 30

model_prefix = "experiments/DA_Faster/LEVIR/model/SSDD_"

commond = "python -u eval/test.py --net {}  --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")