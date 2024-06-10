import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import cv2  # 用于图像处理

# 超参数
batch_size = 256
learning_rate = 0.01
momentum = 0.5
EPOCH = 20

# 准备数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

model = Net()
model.load_state_dict(torch.load('mnist_cnn_old.pth'))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练和测试函数
def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()
        if batch_idx % 300 == 299:  # Print every 300 batches
            acc = 100 * running_correct / running_total
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, acc))
            running_loss = 0.0
            running_total = 0
            running_correct = 0

        

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('Accuracy on test set epoch: %.1f %%' % (100 * acc))
    return acc


# 分割图像中的字符
def segment_characters(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按x坐标排序轮廓
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    character_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        # print(w*h)
        if w*h < 2000: 
            continue
        char_image = binary[y:y+int(1*h), x:x+int(1*w)]
        # char_image = binary[y-int(0.05*h):y+int(1.1*h), x-int(0.05*w):x+int(1.1*w)]
        # 膨胀一下
        kernel = np.ones((5, 5), np.uint8)
        char_image = cv2.dilate(char_image, kernel, iterations=2)
        # char_image 外面包围一圈0   
        char_image = cv2.copyMakeBorder(char_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)


        char_image = cv2.resize(char_image, (28, 28))



        # 反转一下
        # char_image = cv2.bitwise_not(char_image)
        character_images.append(char_image)
        # cv2.imshow('Character', char_image)
        # cv2.waitKey(0)
    return character_images

# 预测单个字符
def predict_character(image):

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    img = Image.fromarray(image)
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        # print(img)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        # print('Predicted:', output)
    return predicted.item()

# 预测图像中的连续数字
def predict_digits(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Image not found or unable to load at path: {image_path}")
    
    characters = segment_characters(image)
    digits = []
    for char in characters:
        digit = predict_character(char)
        digits.append(digit)
    return digits



def train_and_vis(output_model_path='mnist_cnn.pth'):
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)
        torch.save(model.state_dict(), "last_model.pth")
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
    torch.save(model.state_dict(), output_model_path)


def predictDigest(image_path='h_number2.png',model_path='mnist_cnn_old.pth'):

    model.load_state_dict(torch.load(model_path))
    model.eval()

    #预测手写学号图像
    predicted_digits = predict_digits(image_path)
    print('Predicted digits:', predicted_digits)



if __name__ == '__main__':

    # train_and_vis()
    predictDigest(image_path='student_id.jpg')





