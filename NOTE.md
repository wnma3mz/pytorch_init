## 一些Tips

### 关于TF与PyTorch修改梯度

```python
# tensorflow
grads_and_vars = opt.compute_gradients(loss, parameter_list)
my_grads_and_vars = [(g*C, v) for g, v in grads_and_vars]
opt.apply_gradients(my_grads_and_vars)

# pytorch
loss.backward()
for p in model.parameters():
    p.grad *= C  # or whatever other operation
optimizer.step()

# 如果有两个loss，修改其中一个
# retain_graph=True, 确保loss传播完保留。默认为False，为节省显存，在反向传播完之后会清空变量求导
loss2.backward(retain_graph=True) 
for p in model.parameters():
    p.grad *= C
optimizer.step()
optimizer.zero_grad()
loss1.backward()
optimizer.step()
```

```python3
# 梯度修改
weights = model.parameters()
loss.backward()
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
```

### 梯度下降算法，pytorch源码剖析

https://www.rogoso.info/optim-method/

### Numpy to PyTorch

此部分参考https://www.pytorchtutorial.com/pytorch-for-numpy-users/

| Numpy            | PyTorch |
| ---------------- | ------- |
| np.less          | x.lt    |
| np.less_equal    | x.le    |
| np.greater       | x.gt    |
| np.greater_equal | x.ge    |
| np.equal         | x.eq    |
| np.not_equal     | x.ne    |
| x.min       | x.min                          |
| x.argmin    | x.argmin                       |
| x.max       | x.max                          |
| x.argmax    | x.argmax                       |
| x.clip      | x.clamp                        |
| x.round     | x.round                        |
| np.floor(x) | torch.floor(x); x.floor()      |
| np.ceil(x)  | torch.ceil(x); x.ceil()        |
| x.trace     | x.trace                        |
| x.sum       | x.sum                          |
| x.cumsum    | x.cumsum                       |
| x.mean      | x.mean                         |
| x.std       | x.std                          |
| x.prod      | x.prod                         |
| x.cumprod   | x.cumprod                      |
| x.all       | (x == 1).sum() == x.nelement() |
| x.any       | (x == 1).sum() > 0             |
| np.put                                                  |                                                              |
| x.put                                                   | x.put_                                                       |
| x = np.array([1, 2, 3])x.repeat(2) # [1, 1, 2, 2, 3, 3] | x = torch.tensor([1, 2, 3])x.repeat(2) # [1, 2, 3, 1, 2, 3]x.repeat(2).reshape(2, -1).transpose(1, 0).reshape(-1) # [1, 1, 2, 2, 3, 3] |
| np.tile(x, (3, 2))                                      | x.repeat(3, 2)                                               |
| np.choose                                               |                                                              |
| np.sort                                                 | sorted, indices = torch.sort(x, [dim])                       |
| np.argsort                                              | sorted, indices = torch.sort(x, [dim])                       |
| np.nonzero                                              | torch.nonzero                                                |
| np.where                                                | torch.where                                                  |
| x[::-1]                                                 |                                                              |
| x.reshape                              | x.reshape; x.view        |
| x.resize()                             | x.resize_                |
|                                        | x.resize_as_             |
| x.transpose                            | x.transpose or x.permute |
| x.flatten                              | x.view(-1)               |
| x.squeeze()                            | x.squeeze()              |
| x[:, np.newaxis]; np.expand_dims(x, 1) | x.unsqueeze(1)           |
| x.shape   | x.shape      |
| x.strides | x.stride()   |
| x.ndim    | x.dim()      |
| x.data    | x.data       |
| x.size    | x.nelement() |
| x.dtype   | x.dtype      |
| np.array([[1, 2], [3, 4]])                                   | torch.tensor([[1, 2], [3, 4]])                |
| np.array([3.2, 4.3], dtype=np.float16)np.float16([3.2, 4.3]) | torch.tensor([3.2, 4.3], dtype=torch.float16) |
| x.copy()                                                     | x.clone()                                     |
| np.fromfile(file)                                            | torch.tensor(torch.Storage(file))             |
| np.frombuffer                                                |                                               |
| np.fromfunction                                              |                                               |
| np.fromiter                                                  |                                               |
| np.fromstring                                                |                                               |
| np.load                                                      | torch.load                                    |
| np.loadtxt                                                   |                                               |
| np.concatenate                                               | torch.cat                                     |

### Tf to PyTorch

```python
import tensorflow as tf
import torch
import numpy as np

tf.logical_and(a, b)
a & b


tf.is_finite(a)
torch.isfinite(a)


# 这里sh为列表/元组，没有找到torch比较好的实现方式，但numpy是可以转换的
sh = (2,3)
tf.reduce_min(a, sh, keepdims=True)
torch.from_num(np.min(a, sh, keepdims=True))


tf.stop_gradient(a)
a.detach()


# 这里的矩阵乘法tf单独实现了乘之前先转置，但numpy与torch并未实现
tf.matmul(a, b, transpose_a=True)
tf.matmul(a, b, transpose_b=True)
# 如果是三维（多维)数组，不能直接使用.transpose_a()来进行转置，需要额外注意转置的维度。
# 假设是三维数组
torch.matmul(torch.transpose(a, 2, 1), b)
torch.matmul(a, torch.transpose(b, 2, 1))


tf.expand_dims(x, -1)
torch.unsqueeze(x, 1)


tf.slice(x, [0, i, 0], [-1, 1, -1])
x[:, i:i+1, -1]
tf.slice(x, [1, i, 1], [2, 1, 2])
x[1:1+2, i:i+1, 1:1+2]


# 组成对角矩阵，tf分成了两个函数，但是在numpy和torch中都是用.diag直接实现，具体根据输入来输出
# 取出矩阵中的部分对角元素
tf.matrix_diag_part
# 取出矩阵中所有对角元素
tf.matrix_diag
# 根据一维数组的元素，组成对角矩阵
tf.diag
# 如果输入的是一维数组，则等于tf.diag；如果输入的是矩阵，则等同于tf.matrix_diag_part
# 因此如果是多维矩阵，需要重写
torch.diag

# 这里也没找到比较优雅的实现方式
tf.gather_nd(a. ida)
# 根据自己的情况，改写如下，供参考
tmp_lst = []
for line in ida:
    # 取出数据
    tmp = a[][][]
    # 重新写成多维
    tmp_arr = torch.unsqueeze(torch.unsqueeze(tmp, 0), 0)
    tmp_lst.append(tmp_arr)
# 进行拼接
torch.cat(tmp_lst, axis=1)


tf.greater_equal
torch.ge
```
