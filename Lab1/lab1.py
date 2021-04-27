#%%
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork():

    def __init__(self, nn_input_dim, nn_h1_dim, nn_h2_dim, nn_output_dim, learning_rate):
        np.random.seed(0)
        w1 = np.random.randn(nn_h1_dim, nn_input_dim)
        w2 = np.random.randn(nn_h2_dim, nn_h1_dim)
        w3 = np.random.randn(nn_output_dim, nn_h2_dim)
        self.model = {'w1':w1, 'w2':w2, 'w3':w3}
        self.input_dim = nn_input_dim
        self.h1_dim = nn_h1_dim
        self.h2_dim = nn_h2_dim
        self.output_dim = nn_output_dim
        self.learning_rate = learning_rate
        self.loss_a = []
        self.n_a = []


        




    def generate_linear(self, n=30):
        pts = np.random.uniform(0,1,(n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0],pt[1]])

            if pt[0] < pt[1]:
                labels.append(1)
            else:
                labels.append(0)
        return np.array(inputs), np.array(labels).reshape(n,1)

    def print_generate_linear(self):
        print(self.generate_linear())

    def generate_xor_easy(self):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i,0.1*i])
            labels.append(0)
            if i == 5:
                continue
            inputs.append([0.1*i,1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21,1)

   

    def show(self,inputs,labels):
        pred = []
        for j in range(len(labels)):
            pred.append(self.predict(inputs[j]))
            predictions = np.array(pred)
        plt.subplot(1,2,1)
        plt.title('Ground Truth', fontsize=18)
        for i in range(len(labels)):
            if labels[i] == 0:
                plt.plot(inputs[i][0],inputs[i][1], 'ro')
            else:
                plt.plot(inputs[i][0],inputs[i][1], 'bo')
        plt.subplot(1,2,2)
        plt.title('predict', fontsize=18)
        for k in range(len(labels)):
            if predictions[k] <0.5:
                plt.plot(inputs[k][0],inputs[k][1], 'ro')
            else:
                plt.plot(inputs[k][0],inputs[k][1], 'bo')
        plt.show()
        # print(predictions)


    def show_loss(self, loss_a, n_a):
        plt.title('Loss vs Epoch', fontsize=18)
        plt.plot(n_a,loss_a)
        plt.show()

    def show_accuracy(self, inputs, labels):
        pred = []
        for j in range(len(labels)):
            pred.append(self.predict(inputs[j]))
            predictions = np.array(pred)
        count = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                predictions[i] <0.5
                count +=1
            if labels[i] ==1:
                predictions[i] >0.5
                count +=1
        return count/len(labels)



    def predict(self,x):
        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        u1 = w1.dot(x)
        z1 = 1/(1+np.exp(-u1))
        u2 = w2.dot(z1)
        z2 = 1/(1+np.exp(-u2))
        u3 = w3.dot(z2)
        y_pred = 1/(1+np.exp(-u3))
        return y_pred




    def computeLoss(self, labels, predictions):
        loss1 = []
        for i in range(len(labels)):
            loss1.append((np.subtract(labels[i],predictions[i]))**2)
        loss = np.average(loss1)
        # print(loss)
        return loss


    def train(self, traindata, trainlabels,  n_epochs):
        data = traindata
        labels = trainlabels
        
        

        for j in range(n_epochs):
            # print("-->Epoch {} running...".format(i))
            predictions = []
            # random choose data as input
            random_indices = random.sample(range(0,data.shape[0]), data.shape[0])
            data_r = data[random_indices, :]
            labels_r = labels[random_indices, :]
            w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']

            # Forward propogatioin
            for i in range(len(labels)):
                x = data_r[i].reshape(2,1)
                u1 = w1.dot(x)
                z1 = 1/(1+np.exp(-u1))
                u2 = w2.dot(z1)
                z2 = 1/(1+np.exp(-u2))
                u3 = w3.dot(z2)
                y_pred = 1/(1+np.exp(-u3))
                predictions.append(y_pred)

            # Back propogation
                y = labels_r[i]
                dy_p = -2*(y - y_pred)*y_pred*(1-y_pred)
                dw3 = (z2.T)*dy_p                
                dz2 = (w3.T)*dy_p
                ones = np.ones([4,1])
                du2 = np.multiply(dz2,(np.multiply(z2,(ones-z2))))
                dw2 = du2.dot(z1.T)
                dz1 = (w2.T).dot(du2)
                du1 = np.multiply(dz1,(np.multiply(z1,(ones-z1))))
                dw1 = du1.dot(x.T)

            # update parameters
                w1 += -self.learning_rate * dw1
                w2 += -self.learning_rate * dw2
                w3 += -self.learning_rate * dw3

                # print("Update parameters")
                # for k in self.model.keys():
                #     print("{}:{}".format(k,self.model[k]))
                
                # assign new parameters to the model
                self.model = {'w1':w1, 'w2':w2, 'w3':w3}

            predictions = np.array(predictions)
            predictions = predictions.reshape(len(predictions),1)    
            loss = self.computeLoss(labels, predictions)

            
            

            if j % 500 == 0:
                print("Epoch: {} ,Loss: {}".format(j,loss))
                # print(predictions,'\n')
                # print(labels_r)
                self.loss_a.append(loss)
                self.n_a.append(j)






                


            








if __name__ == "__main__":
    learn_r_xor = 0.01
    learn_r_linear = 0.005
    epoch_xor = 4000
    epoch_linear = 500
    net = NeuralNetwork(2,4,4,1,learn_r_linear)
    data, labels = net.generate_linear()

    net.train(data, labels,epoch_linear)
    net.show_loss(net.loss_a, net.n_a)
    net.show(data,labels)
    print('accuracy: {} %'.format(net.show_accuracy(data,labels)*100))
    # print(net.dy_tmp)


    







# %%
