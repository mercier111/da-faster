import os

net = "res101"
part = "test_t"
dataset = "SSDD"
begin_epoch = 1
end_epoch = 5

model_prefix = "experiments/Cylce_Faster_norm+detect_da/SSDD/model/SSDD_"

commond = "python -u train/cycle_faster/test.py --net {}  --cag --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")