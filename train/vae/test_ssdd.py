import os

net = "res101"
part = "test_s"
dataset = "LEVIR"
begin_epoch = 1
end_epoch = 12

model_prefix = "experiments/detect_vae2/LEVIR/model/LEVIR_"

commond = "python -u train/vae/test.py --net {} --cag --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")