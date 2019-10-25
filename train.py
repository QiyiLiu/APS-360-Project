
import os
import torch
import time
import pickle
import argparse
import torch.nn as nn
from CNN import CNN
from RNN import RNN
#from Decoder import RNN
import matplotlib.pyplot as plt
from vocab_build import vocab_build
from torchvision import transforms
from torch.autograd import Variable
from vocab_build import loadcap
from dataloader import dataloader
#from loader import DataLoader, shuffle_data


if __name__ == '__main__':
    #setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=int, default= 1e-5)
    parser.add_argument('-epoch', type=int, default = 100)
    parser.add_argument('-save_iter', type=int, default=2)
    args = parser.parse_args()

    train_dir='train'
    save_iter=args.save_iter
    epochs = args.epoch
    learning_rate=args.lr
    embedding_size = 512
    hidden_size = 512
    gpu_device= None
    # if not running vocab_build.py in advance, run the code below
    cap_dict=loadcap(train_dir)
    vocab=vocab_build(cap_dict, 5) # threshold is 5
    with open(os.path.join(train_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
        print('dictionary stored')

    # if there is a pkl file; open the pickle file
    #with open(os.path.join(train_dir, 'vocab.pkl'), 'rb') as file:
     #   vocab=pickle.load(file)
      #  print('vocab loaded')

    #load data
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    #datapre = DataLoader(train_dir, vocab, transform)
    #data = datapre.gen_data()
    data = dataloader(train_dir, vocab, transform)
    print('train data loaded')

    #set up CNN and RNN
    cnn=CNN(embedding_size= embedding_size)
    rnn=RNN(embedding_dim= embedding_size, hidden_dim= hidden_size, vocab_size= vocab.index)

    #running with CUDA
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_device):
            cnn.cuda()
            rnn.cuda()

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    params= list(cnn.linear.parameters()) + list(rnn.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate)
    losses, iter= [], []
    n=0

    # training
    print('start training')
    for epoch in range(epochs):
        #img_sf, cap_sf = shuffle_data(data, seed= epoch)
        img_sf, cap_sf = data.shuffle(seed=epoch)
        cap_len = len(cap_sf)
        loss_tot=[]
        tic=time.time()
        for i in range(cap_len):
            img_id=img_sf[i]
            image= data.get_img(img_id)
            #image = data.get_image(img_id)
            image = image.unsqueeze(0) #extend input image to 4 dimensional

            #look for CUDA
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_device):
                    image = Variable(image).cuda()
                    cap = torch.cuda.LongTensor(cap_sf[i])
            else:
                image= Variable(image)
                caption = torch.LongTensor(cap_sf[i])

            # if using gpu, delete the two lines below
            #image = Variable(image)
            #caption = torch.LongTensor(cap_sf[i])

            cap_train= caption[:-1] # delete last line
            cnn_out=cnn(image)
            rnn_out=rnn(cnn_out, cap_train)

            loss=criterion(rnn_out, caption)
            loss.backward()
            optimizer.step()
            loss_tot.append(loss)
            cnn.zero_grad()
            rnn.zero_grad()
            n += 1
            print(str(n) + ' iter')
        iter.append(n)
        toc = time.time()
        loss_avg= torch.mean(torch.Tensor(loss_tot))
        losses.append(float(loss_avg))
        print(("Epoch {}: Train loss: {} | time: {}").format(epoch + 1, loss_avg, (toc - tic)))

        if epoch%save_iter ==0:
            torch.save(cnn.state_dict(), os.path.join(train_dir, 'iter_%d_cnn.pkl'%(epoch)))
            print('cnn saved')
            torch.save(rnn.state_dict(), os.path.join(train_dir, 'iter_%d_rnn.pkl'%(epoch)))
            print('rnn saved')

    plt.title("Training Curve")
    plt.plot(iters, losses, label="Loss")
    # plt.plot(iters, train_acc, label="Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.legend(loc='best')
    plt.show()


