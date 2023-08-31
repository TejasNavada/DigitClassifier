if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    import pandas as pd
    from torch import nn


    
    
    
    """
    # converts activations to percentages and instead of the
    highest percent being the predicted label rather the predicted
    label is the percentage above .5, if there is no percentage
    above .5 it automatically considers the prediction innaccurate
    TLDR: only confident prediction contribute to higher accuracy
    def batch_accuracy_50(preds, labels):
        labels = torch.Tensor(labels).type(torch.LongTensor)
        preds = torch.softmax(preds, dim=1)
        correct = preds[range(labels.shape[0]), labels]
        correct = correct>.5
        return correct.float().mean()
    """    
    def batch_accuracy(preds, labels):
        preds = torch.argmax(preds, dim=1)
        return (preds == labels).float().mean()
    def validate_epoch(model):
        with torch.no_grad():
            acc=0
            running_loss = 0
            count = 0
            for xb,yb in val_dl:
                outputs = model(xb)
                acc += batch_accuracy(outputs, yb)
                count+=1
                targets = torch.Tensor(yb).type(torch.LongTensor)
                running_loss += loss_func(outputs, targets).item()
            print("val loss: " + str(round(running_loss,4)))
            accuracy = acc/count
            return round(accuracy.item(), 4)
    
    path = 'train.csv'
    df = pd.read_csv(path)
    number_tensor=torch.Tensor(df.values)
    #seperate data into x and y tensors
    labels = number_tensor[:,0]
    pixels = number_tensor[:,1:].view(len(labels),1,28,28)
    #normalize pixel values
    pixels = pixels/256
    pixels = pixels-.5
    ds = list(zip(pixels,labels))
    # train/val split .8
    train_set,val_set = torch.utils.data.random_split(ds, [33600,8400])
    train_dl = DataLoader(train_set, batch_size=256)
    val_dl = DataLoader(val_set, batch_size=512)        #channels, image size
    CNN_model = nn.Sequential(                      # inputs: 1, 28,28
        nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.BatchNorm2d(4),                                   # 4, 14,14
        nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8),                                   # 8, 7,7
        nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),                                  #16, 4,4
        nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),                                  #32 2,2
        nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),                                  #64 ,1,1
        nn.Conv2d(64, 10, stride=2, kernel_size=3, padding=1),
        nn.Flatten()                                #outputs: 10, 1
    )
    epochs = 30
    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.5,epochs=epochs,steps_per_epoch=len(train_dl), pct_start=0.25, final_div_factor=1e4)
    loss_func = nn.CrossEntropyLoss()
    CNN_model.eval()
    print("val acc: "+str(validate_epoch(CNN_model))) # check accuracy before training (not needed just curious)
    for epoch in range(epochs):
        running_loss = 0
        CNN_model.train()
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            outputs = CNN_model(inputs)
            targets = torch.Tensor(targets).type(torch.LongTensor)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        CNN_model.eval()
        print("epoch: "+str(epoch))
        print("lr: "+str(optimizer.param_groups[0]['lr'])) #see current lr of OneCycle lr scheduler
        print("train loss: "+str(round(running_loss/4, 4)))
        print("val acc: "+str(validate_epoch(CNN_model)))
    

    torch.save(CNN_model.state_dict(), 'models/CNN_hand_number_30.pth')
    


    


