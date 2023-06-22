import numpy as np
import matplotlib.pyplot as plt
# 为了实现mnist的dataset 和dataloader
# 1. 解析所有的training images、labels，test images labels
# 2. 定义dataset，传入的数据是images和labels
# 3. 定义dataloader，此时为每个batch打包images和labels
# 4. 写一个简单的程序，加载并使用dataloader。（区分training和test）

# 1. 是10个数字的分类任务，所以是10个逻辑回归模型
# 2. 正则化的考虑
#    Nx3   -> 3个变量储存mean/std
#    Nx28x28  ->  Nx784
#             ->  784个变量储存mean/std
#                 如果每个数据都是独立分布，则需要独立统计mean/std
#                 对于2d图像，每个pixel并不是独立分布，他们存在空间相关性，因此需要统一考虑mean和std
#                 1个数字储存mean和std
#    对于训练集，需要统计mean和std
#    对于测试集，使用训练集统计的mean和std　
# 3. 我们之前使用sigmoid处理二分类 sigmoid(preidct) -> logits(0-1)
#    在这里，是多分类(10)，我们考虑使用softmax(preidct) -> logits(0-1)
#    你可以自己思考，sigmoid与softmax更本质的区别，比如，一个predict，存在多个预测label时


class MNISTDataset:
    def __init__(self, images_file, labels_file, train, mean=None, std=None):
        self.images = self.parse_mnist_images_file(images_file)
        self.labels = self.parse_mnist_labels_file(labels_file)
        assert len(self.images) == len(self.labels), "Create mnist dataset failed."

        # flatten(Nx28x28) -> Nx784
        self.images = self.images.reshape(len(self.images), -1)

        if train:
            self.images, self.mean, self.std = self.normalize(self.images)
        else:
            self.images, self.mean, self.std = self.normalize(self.images, mean, std)

    @staticmethod
    def normalize(x, mean=None, std=None):
        if mean is None:
            mean = x.mean()

        if std is None:
            std  = x.std()

        x = (x - mean) / std
        return x, mean, std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    # 因为这个函数他是功能性质的，不需要使用到self的任何信息
    # 所以我们可以期待它是一个类的函数，静态方法
    @staticmethod
    def parse_mnist_labels_file(file):
        # 返回值是标签
        with open(file, "rb") as f:
            data = f.read()

        # magic number, num of items
        magic_number, num_of_items = np.frombuffer(data, dtype=">i", count=2, offset=0)
        # 断言
        # 如果cond不满足，则程序跑出assert异常，消息是message
        # assert cond, message
        assert magic_number==2049, "Invalid labels file."
        items = np.frombuffer(data, dtype=np.uint8, count=-1, offset=8).astype(np.int32)
        assert num_of_items==len(items), "Invalid items count."
        return items

    @staticmethod
    def parse_mnist_images_file(file):
        # 返回值是图像
        with open(file, "rb") as f:
            data = f.read()

        # magic number, num of items, rows, columns
        magic_number, num_of_images, rows, columns = np.frombuffer(data, dtype=">i", count=4, offset=0)
        # 断言
        # 如果cond不满足，则程序跑出assert异常，消息是message
        # assert cond, message
        assert magic_number==2051, "Invalid images file."
        pixels = np.frombuffer(data, dtype=np.uint8, count=-1, offset=16)
        images = pixels.reshape(num_of_images, rows, columns)
        return images


class DataloaderIteration:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cursor = 0

        self.indexs = np.arange(len(dataloader.dataset))
        np.random.shuffle(self.indexs)

    def __next__(self):
        # 预期随机抓取一批图像和标签
        begin = self.cursor
        end   = begin + self.dataloader.batch_size
        if end > len(self.dataloader.dataset):
            # 表示已经迭代到结尾了，没有了
            raise StopIteration()

        self.cursor = end
        batched_data = []
        for index in self.indexs[begin:end]:
            batched_data.append(self.dataloader.dataset[index])

        # 对于numpy而言，他可以吧多个ndarray，给拼在一起
        return [np.stack(item, axis=0) for item in list(zip(*batched_data))]

class Dataloader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return DataloaderIteration(self)

training_dataset = MNISTDataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", train=True)
training_dataloader = Dataloader(training_dataset, 32)

test_dataset = MNISTDataset(
    "data/t10k-images.idx3-ubyte",
    "data/t10k-labels.idx1-ubyte",
    train=False,
    mean=training_dataset.mean,
    std=training_dataset.std
)
test_dataloader = Dataloader(test_dataset, 10)


# 定义k和b
# 变量数量 x 输出数量
num_classes = 10
k = np.random.randn(784, num_classes)
b = np.zeros((1, num_classes))

def softmax(x, dim):
    # 输出的概率和为1
    # x(NxM)
    # x = [x0, x1, x2]
    # esum = exp(x0) + exp(x1) + exp(x2)
    # output = [exp(x0)/esum, exp(x1)/esum, exp(x2)/esum]
    # 考虑溢出问题
    # 比如说x = [300, 500, 800]
    # return np.exp(x) / np.exp(x).sum(axis=dim, keepdims=True)
    # 对于x-x.max()是softmax中常见的手段
    x = np.exp(x - x.max())
    return x / x.sum(axis=dim, keepdims=True)

def crossentropy_loss(logits, onehot_labels):
    # logits(Nx10)
    # labels(Nx10)    ->  对于softmax而言，labesl需要转换为onehot
    # onehot ->
    #  3个类别
    #  label = 0    ->   1, 0, 0
    #  label = 1    ->   0, 1, 0
    #  label = 2    ->   0, 0, 1
    batch = logits.shape[0]
    return -(onehot_labels * np.log(logits)).sum() / batch


lr = 1e-2

niter = 0
for epoch in range(10):
    for images, labels in training_dataloader:
        niter += 1

        # images(32x784)
        # labels(32,)
        predict = images @ k + b

        # predict -> logits
        logits = softmax(predict, dim=1)

        # binary crossentropy loss
        # loss = -(y * ln(p) + (1-y) * ln(1-p))

        # softmax crossentropy loss
        # loss = -(y * ln(p))
        batch = logits.shape[0]
        onehot_labels = np.zeros_like(logits)

        # labels(32,)  ->  onehot(32, 10)
        onehot_labels[
            np.arange(batch),
            labels
        ] = 1

        loss = crossentropy_loss(logits, onehot_labels)

        if niter % 100 == 0:
            print(f"Epoch: {epoch}, Iter: {niter}, Lr: {lr:e}, Loss: {loss:.3f}")

        # 自行推导
        G = (logits - onehot_labels) / batch

        # C = AB
        # dA = G @ B.T
        # dB = A.T @ G
        delta_k = images.T @ G
        delta_b = G.sum(axis=0, keepdims=True)

        k = k - lr * delta_k
        b = b - lr * delta_b

    # evaluate
    all_predict = []
    for images, labels in test_dataloader:

        # images(32x784)
        # labels(32,)
        predict = images @ k + b

        # predict -> logits
        logits = softmax(predict, dim=1)

        predict_labels = logits.argmax(axis=1)
        all_predict.extend(predict_labels == labels)

    accuracy = np.sum(all_predict) / len(all_predict) * 100
    print(f"Epoch: {epoch}, Evaluate Test Set, Accuracy: {accuracy:.3f} %")

for images, labels in test_dataloader:

    # images(32x784)
    # labels(32,)
    predict = images @ k + b

    # predict -> logits
    logits = softmax(predict, dim=1)

    predict_labels = logits.argmax(axis=1)

    pixels = (images * training_dataset.std + training_dataset.mean).astype(np.uint8).reshape(-1, 28, 28)
    for image, predict, gt in zip(pixels, predict_labels, labels):
        plt.imshow(image)
        plt.title(f"Predict: {predict}, GT: {gt}")
        plt.show()
