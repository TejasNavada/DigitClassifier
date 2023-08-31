if __name__ == '__main__':
    import torch
    import pandas as pd
    from torch import nn
    
    path = 'test.csv'
    df = pd.read_csv(path)

    model = nn.Sequential(
        nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 10, stride=2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Flatten()
    )

    model.load_state_dict(torch.load('models/CNN_hand_number_30.pth'))
    model.eval()

    number_tensor=torch.Tensor(df.values).view(len(df.values),1,1,28,28)
    number_tensor = number_tensor/256
    number_tensor = number_tensor-.5
    print(number_tensor.shape)
    print(number_tensor[0])
    test_output = torch.Tensor().new_full((number_tensor.size(dim=0), 2), -1)

    for i in range(number_tensor.size(dim=0)):
        test_output[i,0] = i+1  #id
        test_output[i,1] = model(number_tensor[i]).argmax() #label

    px = pd.DataFrame(test_output.numpy().astype(int), columns=['ImageId', 'Label'])
    px.to_csv("test_output.csv",index=False) #save to file



    
